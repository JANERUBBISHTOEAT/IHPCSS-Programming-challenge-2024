/**
 * @file main.c
 * @brief Optimized pagerank implementation with MPI and OpenMP.
 * @author Ludovic Capelli (l.capelli@epcc.ed.ac.uk)
 **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed forthe calculation loop.
#define MAX_TIME 10

/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;

// Cache outdegree to avoid recalculation
int outdegree_cache[GRAPH_ORDER];
 
void initialize_graph(void)
{
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            adjacency_matrix[i][j] = 0.0;
        }
    }
}

/**
 * @brief Precompute outdegree for all vertices to optimize calculation.
 */
void precompute_outdegrees(void)
{
    #pragma omp parallel for schedule(dynamic, 32)
    for(int j = 0; j < GRAPH_ORDER; j++)
    {
        int degree = 0;
        for(int k = 0; k < GRAPH_ORDER; k++)
        {
            if (adjacency_matrix[j][k] == 1.0)
            {
                degree++;
            }
        }
        outdegree_cache[j] = degree;
    }
}

/**
 * @brief Calculates the pagerank of all vertices in the graph with parallel optimization.
 * @param pagerank The array in which store the final pageranks.
 * @param rank The MPI rank of current process.
 * @param size The total number of MPI processes.
 */
void calculate_pagerank(double pagerank[], int rank, int size)
{
    double initial_rank = 1.0 / GRAPH_ORDER;
 
    // Initialise all vertices to 1/n using OpenMP
    #pragma omp parallel for
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        pagerank[i] = initial_rank;
    }
 
    // Precompute outdegrees once
    precompute_outdegrees();
    
    double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
    double diff = 1.0;
    size_t iteration = 0;
    double start = MPI_Wtime();
    double elapsed = 0.0;
    double time_per_iteration = 0;
    
    static double new_pagerank[GRAPH_ORDER];
    static double old_pagerank[GRAPH_ORDER];
    
    #pragma omp parallel for
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        new_pagerank[i] = 0.0;
    }

    // If we exceeded the MAX_TIME seconds, we stop.
    while(elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME)
    {
        // Save old pagerank for comparison
        memcpy(old_pagerank, pagerank, sizeof(double) * GRAPH_ORDER);
 
        // Reset new pagerank array
        #pragma omp parallel for
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            new_pagerank[i] = 0.0;
        }
 
        // Handle dangling nodes (nodes with no outbound edges)
        double dangling_contribution = 0.0;
        #pragma omp parallel for reduction(+:dangling_contribution)
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            if (outdegree_cache[j] == 0)
            {
                dangling_contribution += pagerank[j];
            }
        }
        dangling_contribution /= GRAPH_ORDER;
        
        // Main PageRank calculation with OpenMP parallelization
        #pragma omp parallel for schedule(guided)
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            double sum = 0.0;
            for(int j = 0; j < GRAPH_ORDER; j++)
            {
                if (adjacency_matrix[j][i] == 1.0 && outdegree_cache[j] > 0)
                {
                    sum += pagerank[j] / (double)outdegree_cache[j];
                }
            }
            sum += dangling_contribution;
            new_pagerank[i] = DAMPING_FACTOR * sum + damping_value;
        }
 
        // Calculate convergence metric
        diff = 0.0;
        #pragma omp parallel for reduction(+:diff)
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            diff += fabs(new_pagerank[i] - old_pagerank[i]);
        }
        
        max_diff = (max_diff < diff) ? diff : max_diff;
        total_diff += diff;
        min_diff = (min_diff > diff) ? diff : min_diff;
 
        // Update pagerank
        memcpy(pagerank, new_pagerank, sizeof(double) * GRAPH_ORDER);
            
        // Verification: sum should be 1.0
        double pagerank_total = 0.0;
        #pragma omp parallel for reduction(+:pagerank_total)
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            pagerank_total += pagerank[i];
        }
        if(fabs(pagerank_total - 1.0) >= 1E-12)
        {
            printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n", iteration, pagerank_total);
        }
 
        elapsed = MPI_Wtime() - start;
        iteration++;
        time_per_iteration = elapsed / iteration;
    }
    
    if (rank == 0)
    {
        printf("%zu iterations achieved in %.2f seconds (MPI+OpenMP optimized)\n", iteration, elapsed);
    }
}

/**
 * @brief Populates the edges in the graph for testing with parallel generation.
 **/
void generate_nice_graph(void)
{
    printf("Generate a graph for testing purposes (i.e.: a nice and conveniently designed graph :) )\n");
    double start = omp_get_wtime();
    initialize_graph();
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            if(i != j)
            {
                adjacency_matrix[i][j] = 1.0;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

/**
 * @brief Populates the edges in the graph for the challenge with parallel generation.
 **/
void generate_sneaky_graph(void)
{
    printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n");
    double start = omp_get_wtime();
    initialize_graph();
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER - i; j++)
        {
            if(i != j)
            {
                adjacency_matrix[i][j] = 1.0;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

int main(int argc, char* argv[])
{
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("This program has two graph generators: generate_nice_graph and generate_sneaky_graph. If you intend to submit, your code will be timed on the sneaky graph, remember to try both.\n");
        printf("Running with %d MPI process(es) and OpenMP threads.\n", size);
    }

    // Get the time at the very start
    double start = MPI_Wtime();
    
    // Generate graph on rank 0, then broadcast to other processes
    if (rank == 0)
    {
        generate_nice_graph();
    }
    
    // Broadcast the adjacency matrix to all processes
    MPI_Bcast(adjacency_matrix, 
              GRAPH_ORDER * GRAPH_ORDER, 
              MPI_DOUBLE, 
              0, 
              MPI_COMM_WORLD);
 
    // The array in which each vertex pagerank is stored
    double pagerank[GRAPH_ORDER];
    calculate_pagerank(pagerank, rank, size);
 
    // Only rank 0 prints results
    if (rank == 0)
    {
        // Calculates the sum of all pageranks. It should be 1.0.
        double sum_ranks = 0.0;
        #pragma omp parallel for reduction(+:sum_ranks)
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            sum_ranks += pagerank[i];
        }
        
        // Print sample pageranks
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            if(i % 100 == 0)
            {
                printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
            }
        }
        
        printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", 
               sum_ranks, total_diff, max_diff, min_diff);
        
        double end = MPI_Wtime();
        printf("Total time taken: %.2f seconds.\n", end - start);
    }
    
    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
