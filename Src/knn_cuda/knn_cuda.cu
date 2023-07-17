#include <cstdio>
#include <iostream>
#include <cmath>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <chrono>
#define N 262144 // number of data points
#define D 10// dimension of each data point
#define K 5 // number of nearest neighbors
#define BLOCK_SIZE_DIS 32 // number of threads per block for distance calculation
#define BLOCK_SIZE_MERGE 1 // number of threads per block for merge sort
#define BLOCK_SIZE_BITONIC 1024 // number of threads per block for bitonic sort


//--------------------------------------------------------------------------------------//
//                                     CUDA Kernels                                     //
//--------------------------------------------------------------------------------------//

/**
 * This CUDA kernel calculates the Euclidean distance between a query point and a set of data points.
 *
 * input：dimension, num_points, data_input, query
 * @param dimension The dimension of each point (number of features).
 * @param num_points The number of data points to calculate distances for.
 * @param data_input An array containing the data points.
 * @param query An array containing the query point.
 *
 * output：distances
 * @param distances An array to store the calculated distances.
 *
 */
__global__ void euclidean_distance_kernel(float* distances, float* data_input, float* query, int dimension, int num_points) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float sum = 0;
        for (int i = 0; i < dimension; i++) {
            float diff = data_input[idx * dimension + i] - query[i];
            sum += diff * diff;
        }
        distances[idx] = sqrt(sum);
    }
}

/**
 * This CUDA kernel use a merge sort to sort the distances and indices.
 * It is used in the main function to find the k nearest neighbors.
 *
 * input：num_points, ind_temp, dis_temp
 * @param num_points The number of distances to sort.
 * @param ind_temp An array temprarly storing the indices to pass to the next iteration.
 * @param dis_temp An array temprarly storing the distances to pass to the next iteration.
 *
 * output：indices, distances
 * @param indices An array to store the indices of the sorted distances.
 * @param distances An array to store the sorted distances.
 *
 * @NOTE: The number of threads per block must be 1. And the total number of the data points must be a power of 2.
 */
__global__ void mergeBlocks_kernel(int* indices, float* distances, int* ind_temp, float* dis_temp, int sortedsize) {
    int idx = blockIdx.x;

    //index for distances
    int index1 = idx * 2 * sortedsize;
    int endIndex1 = index1 + sortedsize;
    int index2 = endIndex1;
    int endIndex2 = index2 + sortedsize;
    int targetIndex = idx * 2 * sortedsize;

    //pesudo index for indices
    //let the indices divide and merge like distances
    //int index1_ind = index1;
    //int endIndex1_ind = endIndex1;
    //int index2_ind = index2;
    //int endIndex2_ind = endIndex2;
    //int targetIndex_ind = targetIndex;

    int done = 0;
    while (!done)
    {
        //if the first block is finished and the second block is not finished
        if ((index1 == endIndex1) && (index2 < endIndex2)) {
            dis_temp[targetIndex] = distances[index2];
            ind_temp[targetIndex] = indices[index2];
            targetIndex++;
            index2++;
        }

        //if the second block is finished and the first block is not finished
        else if ((index2 == endIndex2) && (index1 < endIndex1)) {
            dis_temp[targetIndex] = distances[index1];
            ind_temp[targetIndex] = indices[index1];
            targetIndex++;
            index1++;
        }

        //if the first block is smaller than the second block
        else if (distances[index1] < distances[index2]) {
            dis_temp[targetIndex] = distances[index1];
            ind_temp[targetIndex] = indices[index1];
            targetIndex++;
            index1++;
        }

        //else
        else {
            dis_temp[targetIndex] = distances[index2];
            ind_temp[targetIndex] = indices[index2];
            targetIndex++;
            index2++;
        }

        //if both blocks are finished, merge done.
        if ((index1 == endIndex1) && (index2 == endIndex2))
            done = 1;
    }
}

/**
* This CUDA kernel implement a swap function.
*
* input：a, b
* @param a The first number to swap.
* @param b The second number to swap.
*
* output：a, b
* @param a The first number after swapping.
* @param b The second number after swapping.
*
*/
__device__ void swap_float(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}


__device__ void swap_int(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}


/**
* This CUDA kernel implement a bitonic sort.
*
* input：in, length, step, num_points
* @param in An array containing the data points.
* @param length The length that the array is divided into.
* @param step The step that the array is divided into.
* @param num_points The number of data points to sort.
*
* output：in
* @param in An array containing the sorted data points.
*
* @NOTE: The total number of the data points must be a power of 2.
*
*/
__global__ void Bitonic_Sort_kernel(float* in, int* indices, int length, int step, int num_points) {
    unsigned int idx;
    idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < num_points) {
        int index = idx | step; //Find the pair index of the element we want to compare
        if ((idx & length) == 0) { //ascending
            if (in[idx] > in[index]) {
                swap_float(&in[idx], &in[index]);
                swap_int(&indices[idx], &indices[index]);
            }
        }

        else { //descending
            if (in[idx] < in[index]) {
                swap_float(&in[idx], &in[index]);
                swap_int(&indices[idx], &indices[index]);
            }
        }
    }
}



//--------------------------------------------------------------------------------------//
//                                     CPU functions                                    //
//--------------------------------------------------------------------------------------//


/**
 * This CPU function calculates the Euclidean distance between a query point and a set of data points.
 *
 * input：dimension, num_points, data_input, query
 * @param dimension The dimension of each point (number of features).
 * @param num_points The number of data points to calculate distances for.
 * @param data_input An array containing the data points.
 * @param query An array containing the query point.
 *
 * output：distances
 * @param distances An array to store the calculated distances.
 *
 */
float euclidean_distance_cpu(std::vector<float>& vec1, std::vector<float>& vec2) {
    float distance = 0.0;
    for (int j = 0; j < vec1.size(); j++) {
        float diff = vec1[j] - vec2[j];
        distance += diff * diff;
    }

    return sqrt(distance);
}

/**
 * This CPU function implements sorting the distances and indices.
 *
 */
void knn_cpu(std::vector<std::pair<std::vector<float>, float>>* distances, std::vector<std::vector<float>>& trainSet, std::vector<float>& queryData, std::vector<std::pair<float, int>>* indice_sorted) {
    int index = 0;
    for (auto j : trainSet) {
        distances->push_back(std::make_pair(j, euclidean_distance_cpu(queryData, j)));
        indice_sorted->push_back(std::make_pair(euclidean_distance_cpu(queryData, j), index));
        index++;
    }

    std::sort(distances->begin(), distances->end(), [](const std::pair<std::vector<float>, float>& p1, const std::pair<std::vector<float>, float>& p2) {
        return p1.second < p2.second;
        });
    std::sort(indice_sorted->begin(), indice_sorted->end(), [](const std::pair<float, int>& p1, const std::pair< float, int>& p2) {
        return p1.first < p2.first;
        });
}

/**
 *This is a C++ class called "Timer" that allows timing of program execution.
 *It uses the <chrono> library to measure time with high resolution clock.
 *
 */
class Timer {
public:
    Timer() {
        stopped = started = std::chrono::high_resolution_clock::now();
    };

    Timer& start() {
        started = std::chrono::high_resolution_clock::now();
        return *this;
    }

    Timer& stop() {
        stopped = std::chrono::high_resolution_clock::now();
        return *this;
    }

    double elapsed() {
        if (started != stopped) {
            std::chrono::duration<double> elapsed = stopped - started;
            return elapsed.count();
        }
        return 0.0;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> started;
    std::chrono::time_point<std::chrono::high_resolution_clock> stopped;
};

//--------------------------------------------------------------------------------------//
//                                     Main function                                    //
//--------------------------------------------------------------------------------------//


/**
 * main function to see if GPU and CPU functions work and compare running time
 */
int main() {

    /*============== Initialize and Memory Allocated ==============*/

    // (HOST) Define and allocate memory for data input
    float* data_input;
    data_input = (float*)malloc(N * D * sizeof(float));

    // (HOST) Initialize data input
    for (int i = 0; i < N * D; i++) {
        data_input[i] = (float)rand() / (float)RAND_MAX;
    }

    // (HOST) Convert data_input to vector for CPU computation
    std::vector<std::vector<float> > data_input_vec(N, std::vector<float>(D));
    for (int i = 0; i < N * D; i++) {
        data_input_vec[i / D][i % D] = data_input[i];
    }

    /*
    //using this to test if distance is calculated correctly and if the sorting works
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            data_input[i * D + j] = i+1;
        }
    }

    std::vector<float> data_input_vec(data_input, data_input + N * D);
    */

    // (HOST) Define and allocate memory for query data
    float* data_query;
    data_query = (float*)malloc(D * sizeof(float));

    // (HOST) Initialize query data
    for (int i = 0; i < D; i++) {
        data_query[i] = (float)rand() / (float)RAND_MAX;
    }

    // (HOST) Convert data_query to vector for CPU computation
    std::vector<float> data_query_vec(data_query, data_query + D);

    /*
    //using this to test if distance is calculated correctly and if the sorting works
    for (int i = 0; i <D; i++) {
        data_query[i] = -1;
    }
    */

    // (DEVICE) Copy input data to device memory
    float* d_input;
    cudaMalloc(&d_input, N * D * sizeof(float));
    cudaMemcpy(d_input, data_input, N * D * sizeof(float), cudaMemcpyHostToDevice);


    // (DEVICE) Copy query data to device memory
    float* d_query;
    cudaMalloc(&d_query, D * sizeof(float));
    cudaMemcpy(d_query, data_query, D * sizeof(float), cudaMemcpyHostToDevice);

    // (DEVICE) Allocate memory for temporary distances and indices
    float* d_dis_temp;
    cudaMalloc(&d_dis_temp, N * sizeof(float));
    int* d_ind_temp;
    cudaMalloc(&d_ind_temp, N * sizeof(int));

    // (DEVICE) Allocate memory for indices
    int* d_indices_merge;
    int* d_indices_bitonic;
    cudaMalloc(&d_indices_merge, N * sizeof(int));
    cudaMalloc(&d_indices_bitonic, N * sizeof(int));

    /*
    // Allocate memory for unsorted distances
    float* distances_unsorted = (float*)malloc(N * sizeof(float));
    cudaMemcpy(distances_unsorted, d_distances, N * sizeof(float), cudaMemcpyDeviceToHost);
    */

    // (DEVICE) Initialize indices and copy to device memory
    int* indices = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        indices[i] = i;
    }
    cudaMemcpy(d_indices_merge, indices, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices_bitonic, indices, N * sizeof(int), cudaMemcpyHostToDevice);

    // (DEVICE) Allocate memory for distances
    float* d_distances_merge;
    float* d_distances_bitonic;
    cudaMalloc(&d_distances_merge, N * sizeof(float));
    cudaMalloc(&d_distances_bitonic, N * sizeof(float));

    // (HOST) Allocate memory for sorted distances and indices computed by GPU
    float* distances_sorted_merge = (float*)malloc(N * sizeof(float));
    float* distances_sorted_bitonic = (float*)malloc(N * sizeof(float));
    int* indices_sorted_merge = (int*)malloc(N * sizeof(int));
    int* indices_sorted_bitonic = (int*)malloc(N * sizeof(int));

    // (HOST) Allocate memory for sorted distances and indices computed by CPU
    //float* distances_sorted_CPU = (float*)malloc(N * sizeof(float));
    auto indices_sorted_CPU_vec = new std::vector<float>();


    /*============== GPU functions ==============*/

    /*-----Merge Sort-----*/
    cudaEvent_t start1, stop1;
    checkCudaErrors(cudaEventCreate(&start1));
    checkCudaErrors(cudaEventCreate(&stop1));

    checkCudaErrors(cudaEventRecord(start1));

    // Compute distances between current query point and data input
    euclidean_distance_kernel << <(N + BLOCK_SIZE_DIS - 1) / BLOCK_SIZE_DIS, BLOCK_SIZE_DIS >> >
        (d_distances_merge, d_input, d_query, D, N);

    // Sort distances and indices
    int blocks_merge = (N + BLOCK_SIZE_MERGE - 1) / BLOCK_SIZE_MERGE / 2;
    int sortedsize = BLOCK_SIZE_MERGE;
    while (blocks_merge > 0) {
        mergeBlocks_kernel << <blocks_merge, 1 >> > (d_indices_merge, d_distances_merge, d_ind_temp, d_dis_temp, sortedsize);
        cudaMemcpy(d_distances_merge, d_dis_temp, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_indices_merge, d_ind_temp, N * sizeof(int), cudaMemcpyDeviceToDevice);
        blocks_merge /= 2;
        sortedsize *= 2;
    }

    checkCudaErrors(cudaEventRecord(stop1));

    // Copy sorted distances and indices to host memory
    cudaMemcpy(indices_sorted_merge, d_indices_merge, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances_sorted_merge, d_distances_merge, N * sizeof(float), cudaMemcpyDeviceToHost);

    //    std::vector<float> distances_sorted_vec_GPU(distances_sorted, distances_sorted + N);

    checkCudaErrors(cudaEventSynchronize(stop1));

    /*-----Bitonic Sort-----*/
    cudaEvent_t start2, stop2;
    checkCudaErrors(cudaEventCreate(&start2));
    checkCudaErrors(cudaEventCreate(&stop2));

    checkCudaErrors(cudaEventRecord(start2));


    // Compute distances between current query point and data input
    euclidean_distance_kernel << <(N + BLOCK_SIZE_DIS - 1) / BLOCK_SIZE_DIS, BLOCK_SIZE_DIS >> >
        (d_distances_bitonic, d_input, d_query, D, N);

    int blocks_bitonic = (N + BLOCK_SIZE_BITONIC - 1) / BLOCK_SIZE_BITONIC;
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            Bitonic_Sort_kernel << <blocks_bitonic, BLOCK_SIZE_BITONIC >> > (d_distances_bitonic, d_indices_bitonic, k, j, N);
        }
    }

    checkCudaErrors(cudaEventRecord(stop2));

    // Copy sorted distances and indices to host memory
    cudaMemcpy(indices_sorted_bitonic, d_indices_bitonic, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances_sorted_bitonic, d_distances_bitonic, N * sizeof(float), cudaMemcpyDeviceToHost);

    //    std::vector<float> distances_sorted_vec_GPU(distances_sorted, distances_sorted + N);

    checkCudaErrors(cudaEventSynchronize(stop2));

    printf("============ Settings ============\n\n");
    // Print query point
    printf("Query point : ");
    for (int i = 0; i < D; i++) {
        printf("%f ", data_query[i]);
    }
    printf("\n\n");

    printf("number of data points : %d\n", N);
    printf("number of dimensions : %d\n", D);
    printf("number of K : %d\n", K);
    printf("\n\n");

    printf("============== GPU ==============\n\n");

    // Compute and print kernel execution time for merge sort
    float kernel_time1;
    checkCudaErrors(cudaEventElapsedTime(&kernel_time1, start1, stop1));
    printf("Kernel execution time of merge sort\t\t\t: %f ms\n\n", kernel_time1);



    // Print sorted indices and distances
    printf("Find the indices of K nearest neighbors(GPU, merge) : \n");
    for (int j = 0; j < K; j++) {
        printf("%d ", indices_sorted_merge[j]);
    }
    printf("\n\n");

    printf("Find the distances of K nearest neighbors(GPU, merge) : \n");
    for (int j = 0; j < K; j++) {
        printf("%f ", distances_sorted_merge[j]);
    }
    printf("\n\n");

    printf("---------------------------------\n\n");

    // Compute and print kernel execution time for bitonic sort
    float kernel_time2;
    checkCudaErrors(cudaEventElapsedTime(&kernel_time2, start2, stop2));
    printf("Kernel execution time of bitonic sort\t\t\t: %f ms\n\n", kernel_time2);

    // Print sorted indices and distances
    printf("Find the indices of K nearest neighbors(GPU, bitonic) : \n");
    for (int j = 0; j < K; j++) {
        printf("%d ", indices_sorted_bitonic[j]);
    }
    printf("\n\n");

    printf("Find the distances of K nearest neighbors(GPU, bitonic) : \n");
    for (int j = 0; j < K; j++) {
        printf("%f ", distances_sorted_bitonic[j]);
    }
    printf("\n\n");


    /*============== CPU functions ==============*/

    auto started = std::chrono::high_resolution_clock::now();
    auto distances_sorted_CPU = new std::vector<std::pair<std::vector<float>, float>>();
    auto indices_sorted_CPU = new std::vector<std::pair<float, int>>();
    knn_cpu(distances_sorted_CPU, data_input_vec, data_query_vec, indices_sorted_CPU);

    auto stopped = std::chrono::high_resolution_clock::now();

    /*============== Compare result ==============*/

    printf("============== CPU ==============\n\n");

    std::chrono::duration<double, std::milli> elapsed = stopped - started;
    std::cout << "CPU execution time\t\t\t:" << elapsed.count() << " ms\n\n";

    // Print sorted indices and distances
    printf("Find the indices of K nearest neighbors(CPU) : \n");
    for (int j = 0; j < K; j++) {
        printf("%d ", indices_sorted_CPU->at(j).second);
    }
    printf("\n\n");

    printf("Find the distances of K nearest neighbors(CPU) : \n");
    for (int j = 0; j < K; j++) {
        printf("%f ", distances_sorted_CPU->at(j).second);
    }
    printf("\n\n");
    //

    /*============== Free memory ==============*/

    checkCudaErrors(cudaEventDestroy(start1));
    checkCudaErrors(cudaEventDestroy(stop1));
    checkCudaErrors(cudaEventDestroy(start2));
    checkCudaErrors(cudaEventDestroy(stop2));

    // Free memory
    free(data_input);
    free(data_query);
    //free(indices_sorted);
    free(indices_sorted_merge);
    free(indices_sorted_bitonic);
    //free(distances_sorted);
    free(distances_sorted_merge);
    free(distances_sorted_bitonic);
    free(distances_sorted_CPU);
    //free(indices_sorted_CPU);
    free(indices);
    //free(distances_unsorted);
    cudaFree(d_dis_temp);
    cudaFree(d_ind_temp);
    cudaFree(d_query);
    cudaFree(d_input);
    //cudaFree(d_distances);
    cudaFree(d_distances_merge);
    cudaFree(d_distances_bitonic);
    cudaFree(d_indices_merge);
    cudaFree(d_indices_bitonic);

    return 0;
}
