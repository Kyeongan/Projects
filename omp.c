/*-----------------------------------------------
*	Desc   : k-means functaion
*          - Multithread programming Using OpenMP
*
*	Last Updated : 2011-12-07
------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <omp.h>
#include "sys/time.h"

/* function prototupes */
int **k_means(double ***data, int height, int width, int m);

void allocate_2D_int_matrix(int ***matrix, int dim0, int dim1);
void free_2D_int_matrix(int ***matrix);
void allocate_2D_double_matrix(double ***matrix, int dim0, int dim1);
void free_2D_double_matrix(double ***matrix);
void free_3D_double_matrix(double ****matrix);
void allocate_3D_double_matrix(double ****matrix, int dim0, int dim1, int dim2);

/* constants */
#define NUM_CHANNELS 24

/*-----------------------------------------------
*	Grobal Variable
------------------------------------------------*/
int cnt_proc, rank, rc, numThread, threadId = 0; // Count of Process, Rank(ID), Return Code
double start_time, end_time, elapse_time = 0;	// For the mesurement of time.

int main(int argc, char **argv)
{
	int height, width;
	int **labels = NULL;
	FILE *fpa;
	double ***data;
	/*
  ** Step 1: determine what we have to do
*/
	if (argc != 5)
	{
		printf("usage: %s <inputfile> <outputfile> <height> <width>\n", argv[0]);
		exit(0);
	}

	/* argv[1] is the number of threads */
	/* argv[2] is name of the input file name */
	/* argv[2] is name of the output file name */
	/* argv[3] is height of image */
	/* argv[4] is width of image */

	//	numThread = atoi (argv[1]);
	height = atoi(argv[3]);
	width = atoi(argv[4]);

#pragma omp parallel
	numThread = omp_get_num_threads();
	printf("Threads Number [%d]\n", numThread);
	printf("Input Size [%d][%d]\n", height, width);

	/*
	** Step 2: read the image from file
	*/
	allocate_3D_double_matrix(&data, height, width, NUM_CHANNELS);
	fpa = fopen(argv[1], "rb");
	fread(&(data[0][0][0]), sizeof(double), height * width * NUM_CHANNELS, fpa);
	fclose(fpa);

	double start;
	double end;
	start = omp_get_wtime();

	/*
	** Step 3: Perform k_means clustering over the data items
	*/
	labels = k_means(data, height, width, NUM_CHANNELS);

	// stop timer
	end = omp_get_wtime();
	printf("Work took %f sec. time.\n", end - start);

	/*
	** Step 4:  write final output file
	*/
	printf("Writing final result\n");
	fpa = fopen(argv[2], "wb");
	fwrite(&(labels[0][0]), sizeof(int), height * width, fpa);
	fclose(fpa);

	free_2D_int_matrix(&labels);
	free_3D_double_matrix(&data);
	return 0;
}

int **k_means(double ***data, int height, int width, int nchannels)
{
	int n = height * width;
	int nclusters = 5;				   /* number of clusters to be formed */
	double t = 1e-04;				   /* required precision */
	int h, h1, h2, i, j;			   /* loop counters */
	double old_error, error = DBL_MAX; /* sum of squared euclidean distance */

	double **c = NULL;
	double **c1 = NULL; /* temp centroids */

	int *counts = NULL;  /* size of each cluster */
	int **labels = NULL; /* output cluster label for each data point */

	// for Parallization
	int *local_counts = NULL;
	double local_error = DBL_MAX;

	int dims[2] = {nclusters, nchannels}; // {5, 24}

	allocate_2D_int_matrix(&labels, height, width);
	memset(&(labels[0][0]), 0, height * width * sizeof(int));
	counts = (int *)calloc(nclusters, sizeof(int));
	local_counts = (int *)calloc(nclusters, sizeof(int));

	if (NULL == counts || NULL == labels || NULL == local_counts)
	{
		printf("k_means: error allocating memory\n");
		exit(-1);
	}

	/* for debugging */
	assert(data && nclusters > 0 && nclusters <= n && nchannels > 0 && t >= 0);

	/* initialization */
	allocate_2D_double_matrix(&c1, dims[0], dims[1]);
	allocate_2D_double_matrix(&c, dims[0], dims[1]);

	/* pick nclusters (5) points as initial centroids */
	for (h = i = 0; i < nclusters; h += width / nclusters, i++)
	{
		for (j = 0; j < nchannels; j++)
		{
			c[i][j] = data[h / width][h % width][j];
		}
	}

	double min_distance;
	double distance;

	/* main loop */
	do
	{
		/* save error from last step */
		old_error = error;
		error = 0;

#pragma omp parallel for private(i, j) firstprivate(nclusters, nchannels, counts)
		/* clear old counts and temp centroids */
		for (i = 0; i < nclusters; i++)
		{
			counts[i] = 0;
			for (j = 0; j < nchannels; j++)
			{
				c1[i][j] = 0;
			}
		}

#pragma omp parallel for private(h1, h2, i, j, min_distance, distance) \
	firstprivate(height, width, nclusters, nchannels, labels, c)       \
		shared(counts, data, c1)                                       \
			schedule(static)                                           \
				reduction(+                                            \
						  : error)

		for (h1 = 0; h1 < height; h1++)
		{
			for (h2 = 0; h2 < width; h2++)
			{
				/* identify the closest cluster */
				min_distance = DBL_MAX;
				for (i = 0; i < nclusters; i++) // nclusters (5 centriods)
				{
					distance = 0;
					for (j = 0; j < nchannels; j++)
					{
						distance += pow(data[h1][h2][j] - c[i][j], 2);
					}
					if (distance < min_distance)
					{
						labels[h1][h2] = i;
						min_distance = distance;
					}
				}

				/* update size and temp centroid of the destination cluster */
				for (j = 0; j < nchannels; j++)
				{
#pragma omp atomic
					c1[labels[h1][h2]][j] += data[h1][h2][j];
				}
#pragma omp atomic
				counts[labels[h1][h2]]++;

#pragma omp atomic
				/* update standard error */
				error += min_distance;

			} // width loop
			  //printf("My thread number =[%d]\n", omp_get_thread_num());

		} // height loop - end of the pragma parallel

		// main thread do this.
		/* update all centroids */
		for (i = 0; i < nclusters; i++)
		{
			for (j = 0; j < nchannels; j++)
			{
				c[i][j] = counts[i] ? c1[i][j] / counts[i] : c1[i][j]; // sequencial version
			}
		}

		printf("Error is: %lf\n", error);
	} while (fabs(error - old_error) > t);

	/* Memory Management */
	free_2D_double_matrix(&c1);
	free_2D_double_matrix(&c);
	free(counts);

	return labels;
}

void free_2D_int_matrix(int ***matrix)
{
	int *matrix_tmp0;
	int **matrix_tmp1;

	matrix_tmp0 = **matrix;
	matrix_tmp1 = *matrix;

	free(matrix_tmp0);
	free(matrix_tmp1);

	*matrix = NULL;
	return;
}

void allocate_2D_int_matrix(int ***matrix, int dim0, int dim1)
{
	int i;
	int **tmp_field0;
	int *data;

	data = (int *)malloc(dim0 * dim1 * sizeof(int));
	if (data == NULL)
	{
		printf("%d @ %s, Could not allocate memory\n", __LINE__, __FILE__);
		return;
	}

	tmp_field0 = (int **)malloc(dim0 * sizeof(int *));
	if (tmp_field0 == NULL)
	{
		printf("%d @ %s, Could not allocate memory\n", __LINE__, __FILE__);
		return;
	}
	for (i = 0; i < dim0; i++)
	{
		tmp_field0[i] = &(data[i * dim1]);
	}

	*matrix = tmp_field0;
	return;
}

void free_2D_double_matrix(double ***matrix)
{
	double *matrix_tmp0;
	double **matrix_tmp1;

	matrix_tmp0 = **matrix;
	matrix_tmp1 = *matrix;

	free(matrix_tmp0);
	free(matrix_tmp1);

	*matrix = NULL;
	return;
}

void allocate_2D_double_matrix(double ***matrix, int dim0, int dim1)
{
	int i;
	double **tmp_field0;
	double *data;

	data = (double *)malloc(dim0 * dim1 * sizeof(double));
	if (data == NULL)
	{
		printf("%d @ %s, Could not allocate memory\n", __LINE__, __FILE__);
		return;
	}

	tmp_field0 = (double **)malloc(dim0 * sizeof(double *));
	if (tmp_field0 == NULL)
	{
		printf("%d @ %s, Could not allocate memory\n", __LINE__, __FILE__);
		return;
	}
	for (i = 0; i < dim0; i++)
	{
		tmp_field0[i] = &(data[i * dim1]);
	}

	*matrix = tmp_field0;
	return;
}

void allocate_3D_double_matrix(double ****matrix, int dim0, int dim1, int dim2)
{
	int i;
	double ***tmp_field1;
	double **tmp_field0;
	double *data;

	data = (double *)malloc(dim0 * dim1 * dim2 * sizeof(double));
	if (data == NULL)
	{
		printf("%d @ %s, Could not allocate memory\n", __LINE__, __FILE__);
		exit(-1);
	}

	tmp_field0 = (double **)malloc(dim0 * dim1 * sizeof(double *));
	if (tmp_field0 == NULL)
	{
		printf("%d @ %s, Could not allocate memory\n", __LINE__, __FILE__);
		exit(-1);
	}
	for (i = 0; i < (dim0 * dim1); i++)
	{
		tmp_field0[i] = &(data[i * dim2]);
	}
	tmp_field1 = (double ***)malloc(dim0 * sizeof(double **));
	if (tmp_field1 == NULL)
	{
		printf("%d @ %s, Could not allocate memory\n", __LINE__, __FILE__);
		exit(-1);
	}
	for (i = 0; i < dim0; i++)
	{
		tmp_field1[i] = &(tmp_field0[i * dim1]);
	}

	*matrix = tmp_field1;
	return;
}

void free_3D_double_matrix(double ****matrix)
{
	double *matrix_tmp0;
	double **matrix_tmp1;
	double ***matrix_tmp2;

	matrix_tmp0 = ***matrix;
	matrix_tmp1 = **matrix;
	matrix_tmp2 = *matrix;

	free(matrix_tmp0);
	free(matrix_tmp1);
	free(matrix_tmp2);

	*matrix = NULL;
	return;
}
