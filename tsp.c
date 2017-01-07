#include <float.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int n;
char file_path[255];
int compStats = 0;
int local_rank, numberOfProcesses;

// Timing
double generationTime, processingTime, communicationTime, totalTime;

typedef enum {
  linear,
  nearestNeighbor
} TYPE;
TYPE type;

typedef struct {
  int lineNum;
  float col1;
  float col2c;
  int isVisited;
} LOCATION;
LOCATION *fileLocations;

typedef struct {
  float cost;
  int firstPoint;
  int lastPoint;
} COST;


int parseArguments(int argc, char **argv);
void parseFile();

void swapValues(int *p1, int *p2);
void performLinearOperation();
void performNearestNeighborOp();
void printFinalPath(int *a, int *count, float cost);
float evaluateCost(int *a);
float dist(LOCATION a, LOCATION b);
int findNearest(int current, int start, int end);

int main(int argc, char **argv) {
	double t_start, t_end;
  int i;

  generationTime = 0.0; processingTime = 0.0; communicationTime = 0.0; totalTime = 0.0;

  // Parse the arguments
  if( parseArguments(argc, argv) ) return 1;
  parseFile();
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

  t_start = MPI_Wtime();

  if( type == linear ) {
    performLinearOperation();
  } else if( type == nearestNeighbor ) {
    performNearestNeighborOp();
  }

  t_end = MPI_Wtime();
  totalTime = t_end - t_start;

  if( compStats ) {
    printf("%d\tg\t%d\t%d\t%f\n", n, local_rank, numberOfProcesses, generationTime);
    printf("%d\tp\t%d\t%d\t%f\n", n, local_rank, numberOfProcesses, processingTime);
    printf("%d\tc\t%d\t%d\t%f\n", n, local_rank, numberOfProcesses, communicationTime);
    printf("%d\tt\t%d\t%d\t%f\n", n, local_rank, numberOfProcesses, totalTime);
  }

  free(fileLocations);
	MPI_Finalize(); // Exit MPI
	return 0;
}

void printFinalPath(int *a, int *count, float cost) {
  int x;
  for( x = 0; x < n; x++ )
    printf("%d  ",a[x]);
  printf("cost:%f count:%d\n", cost, *count);
}

void swapValues(int *p1, int *p2) {
  int temp;
  temp = *p1;
  *p1 = *p2;
  *p2 = temp;
}

float dist(LOCATION a, LOCATION b) {
  return sqrt(pow(a.col1 - b.col1, 2) + pow(a.col2c - b.col2c, 2));
}

float evaluateCost(int *a) {
  int i;
  float cost = 0.0f;
  for( i = 0; i < n - 1; i++ ) {
    cost += dist(fileLocations[a[i] - 1], fileLocations[a[i + 1] - 1]);
  }

  return cost;
}

void performLinearOperation() {
  int count, i, x, y;
  int n_perm = 1;
  int *a;
  float cost = 0.0;;
  double start, end, dt;

  start = MPI_Wtime();

  for( i = 1; i <= n; i++ ) {
    n_perm *= i;
  }

  a = (int*) malloc(sizeof(int) * n * n_perm);
  for( i = 0; i < n; i++ ) a[i] = i + 1;

  while( count < n_perm ) {
    for( y = 0; y < n - 1; y++) {
      swapValues(&a[y], &a[y + 1]);
      cost = evaluateCost(a);
      if( !compStats) printFinalPath(a, &count, cost);
      count++;
    }
    swapValues(&a[0], &a[1]);
    cost = evaluateCost(a);
    if( !compStats) printFinalPath(a, &count, cost);
    count++;

    for( y = n - 1; y > 0; y-- ) {
      swapValues(&a[y], &a[y - 1]);
      cost = evaluateCost(a);
      if( !compStats) printFinalPath(a, &count, cost);
      count++;
    }
    swapValues(&a[n - 1], &a[n - 2]);
    cost = evaluateCost(a);
    if( !compStats) printFinalPath(a, &count, cost);
    count++;
  }
  end = MPI_Wtime();
  dt = end - start;
  processingTime += dt;
}

int findNearest(int current, int start, int end) {
  int i, index;
  float min = FLT_MAX;
  float distance;

  index = -1;
  for( i = start; i<= end; ++i ) {
    distance = dist(fileLocations[current], fileLocations[i]);
    if( distance < min && i != current && fileLocations[i].isVisited == 0 ) {
      min = distance;
      index = i;
    }
  }

  return index;
}

void performNearestNeighborOp() {
  int i, j, index;
  float cost = 0.0f;
  int starting_loc, ending_loc;
  int loc_per_node = n / numberOfProcesses;
  int next;
  int *index_of_min;
  float distance;
  float min = FLT_MAX;
  int final_path[n];
  double start, end, dt;


  index_of_min = (int*)malloc(sizeof(int) * numberOfProcesses);
  starting_loc = loc_per_node * local_rank;
  ending_loc = starting_loc + loc_per_node - 1;
  if( local_rank == numberOfProcesses - 1 ) ending_loc += n % numberOfProcesses;

  next = 0;
  final_path[0] = 0;
  for( i = 0; i < n - 1; i++ ) {
    // Find the nearest neighbor to
    start = MPI_Wtime();
    MPI_Bcast(&next, 1, MPI_INT, 0, MPI_COMM_WORLD);
    end = MPI_Wtime();
    dt = end - start;
    communicationTime += dt;

    start = MPI_Wtime();
    fileLocations[next].isVisited = 1;
    int index = findNearest(next, starting_loc, ending_loc);
    end = MPI_Wtime();
    dt = end - start;
    processingTime += dt;

    start = MPI_Wtime();
    MPI_Gather(&index, 1, MPI_INT, index_of_min, 1, MPI_INT, 0, MPI_COMM_WORLD);
    end = MPI_Wtime();
    dt = end - start;
    communicationTime += dt;

    if( local_rank == 0 ) {
      start = MPI_Wtime();
      index = index_of_min[0];
      // find the nearest
      min = FLT_MAX;
      for( j = 0; j < numberOfProcesses; ++j ) {
        if( index_of_min[j] < 0 ) continue;
        distance = dist(fileLocations[next], fileLocations[index_of_min[j]]);
        if( distance < min ) {
          min = distance;
          index = index_of_min[j];
        }
      }
      next = index;
      final_path[i + 1] = index;
      end = MPI_Wtime();
      dt = end - start;
      processingTime += dt;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if( local_rank == 0 && !compStats ) {
    for( i = 0; i < n; ++i ) {
      printf("%d ", final_path[i]);
    }
    printf("\n");
  }
  free(index_of_min);
}

void parseFile() {
  FILE *fp;
  int i, line;
  char buffer[1024];
  float x, y;

  fp = fopen(file_path, "r");

  for(i = 0; i < 7; i++) {
    fgets(buffer, 1024, fp);
  }

  while(fscanf(fp, "%d %f %f", &line, &x, &y) > 0 ) {
    if(line == n) break;
    //printf("%d %f %f\n", line, x, y);
  }

  // Line has the number of elements
  fileLocations = (LOCATION *) malloc(sizeof(LOCATION) * line);
  rewind(fp);

  for(i = 0; i < 7; i++) {
    fgets(buffer, 1024, fp);
  }

  while(fscanf(fp, "%d %f %f", &line, &x, &y) > 0 ) {
    fileLocations[line - 1].lineNum = line;
    fileLocations[line - 1].col1 = x;
    fileLocations[line - 1].col2c = y;
    fileLocations[line - 1].isVisited = 0;

    if(line == n) break;
    //printf("%d %f %f\n", locations[line - 1].line, locations[line - 1].x, locations[line - 1].y);
  }

  fclose(fp);
}

int parseArguments(int argc, char **argv) {
	int i, c;
  int option_index = 0;
  static struct option long_options[] =
  {
    {"input_method1", required_argument,  0, 'q'},
    {"input_method2", required_argument,  0, 'w'},
    {0, 0, 0, 0}
  };

  char *result = NULL;
  char delims[] = "m";
	while( (c = getopt_long (argc, argv, "f:n:t:c", long_options, &option_index)) != -1 ) {
		switch(c) {
      case 'f':
        strcpy(file_path, optarg);
        break;
			case 'n':
				n = atoi(optarg);
				break;
      case 'c':
        compStats = 1;
        break;
      case 't':
        if( strcmp(optarg, "linear" ) == 0 ) type = linear;
        else if( strcmp(optarg, "nn" ) == 0 ) type = nearestNeighbor;
        else {
          fprintf( stderr, "Option -%c %s in incorrect. Allowed values are: linear, nn\n", optopt, optarg);
          return 1;
        }
        break;
      case '?':
				if( optopt == 'n' )
					fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				else if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
				return 1;
			default:
				fprintf(stderr, "Usage: %s -n <number of numbers> \n", argv[0]);
				fprintf(stderr, "\tExample: %s -n 1000\n", argv[0]);
				return 1;
		}
	}
	return 0;
}
