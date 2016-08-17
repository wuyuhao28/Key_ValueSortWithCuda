
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <helper_cuda.h>

#include <time.h>
#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

#define MaxStringLen 16                    //实际可用长度-1
#define KeyNum 3
#define DataLine 7 
//#define DataLine 262144 //1024 * 256
//#define DataLine 98304  //1024 * 96 
//#define DataLine 1048576  //1024 * 1024 

const char path[100] = "TestData.txt";
const char path2[100] = "Data.txt";

struct KeyString
{
	char key[KeyNum][MaxStringLen];
};

bool __device__ __host__ operator < (KeyString l, KeyString r)
{
	for (int k = 0; k < KeyNum; k++)
	{
		for (long int i = 0; i < MaxStringLen; i++)
		{
			if (l.key[k][i] == '\0' && r.key[k][i] == '\0')    //into next key
			{
				if (k == KeyNum - 1)
					return false;
				break;
			}

			if (l.key[k][i] != '\0' && r.key[k][i] == '\0')
				return false;

			if (l.key[k][i] == '\0' && r.key[k][i] != '\0')
				return true;

			if (l.key[k][i] < r.key[k][i])
				return true;

			if (l.key[k][i] == r.key[k][i])
				continue;

			//if l >= r
			return false;                  
		}
	}
}

int main()
{
	const int key_num = KeyNum;
	const int dataline = DataLine;

	void GenerateRandomData(const int key_num, const int dataline);
	GenerateRandomData(key_num, dataline);

	//cudaDeviceProp devprop;
	int deviceID = -1;
	if (cudaSuccess == cudaGetDevice(&deviceID))
	{
		printf("deviceID: %d\n", deviceID);
		cudaDeviceProp devprop;
		cudaGetDeviceProperties(&devprop, deviceID);
		long int total_memory;
		total_memory = sizeof(char) * KeyNum * MaxStringLen * DataLine + sizeof(long int) * DataLine;
		printf("%ld bytes memory used, %ld bytes total memory available.\n", total_memory, devprop.totalGlobalMem);
		if ( devprop.totalGlobalMem < total_memory )
		{
			printf("Error: insufficient amount of GPU memory.\n");
		}
	}

	thrust::host_vector<KeyString> key_vector(DataLine);
	thrust::host_vector<long int> value_vector(DataLine);

	FILE *fp;
	fp = fopen(path, "r+");                            //path做功能测试，path2做大量数据测试
	//fp = fopen(path2, "r+");
	for (long int i = 0; i < DataLine; i++)
	{

		for (int j = 0; j < key_num; j++)
		{
			fscanf(fp, "%s", key_vector[i].key[j]);
		}
		fscanf(fp, "%ld", &value_vector[i]);

		//printf("%s %s %s %ld\n", key_vector[i].key[0], key_vector[i].key[1], key_vector[i].key[2], value_vector[i]);
	}
	fclose(fp);

	printf("vector.size(): %ld\n", key_vector.size());
	printf("vector.capacity(): %ld\n", key_vector.capacity());

	cudaEvent_t start_event, stop_event;

    cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	thrust::device_vector<KeyString> dkey_vector(DataLine);
	thrust::device_vector<long int> dvalue_vector(DataLine);

	cudaEventRecord(start_event, 0);
	dkey_vector = key_vector;
	dvalue_vector = value_vector;

	printf("dvector.size(): %ld\n", dkey_vector.size());
	printf("dvector.capacity(): %ld\n", dkey_vector.capacity());

	thrust::sort_by_key(dkey_vector.begin(),dkey_vector.end(),dvalue_vector.begin());

	key_vector = dkey_vector;
	value_vector = dvalue_vector;

	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);

	for (long int i = 0; i < DataLine; i++)
	{
		for (int j = 0; j < key_num; j++)
		{
			printf("%s\t", key_vector[i].key[j]);
		}
		printf("%ld\n", value_vector[i]);

		if ( i > 100 )
		{
			break;
		}
	}

	float time = 0;
	cudaEventElapsedTime(&time, start_event, stop_event);
	time /= 1.0e3f;
	printf("\nGPU ElapsedTime: %.5f s\n", time);

	//cudaFree(dkey_vector);
	// Free vector memeory, 创建临时空间以释放内存
	key_vector.clear();
	value_vector.clear();
	dkey_vector.clear();
	dvalue_vector.clear();
	thrust::host_vector<KeyString>(key_vector).swap(key_vector);
	thrust::host_vector<long int>(value_vector).swap(value_vector);
	thrust::device_vector<KeyString>(dkey_vector).swap(dkey_vector);
	thrust::device_vector<long int>(dvalue_vector).swap(dvalue_vector);

	return 0;
}

__host__ void GenerateRandomData(const int key_num, const int dataline)
{
	FILE *fp;
	fp = fopen(path2, "w+");
	thrust::default_random_engine rng(clock());
	thrust::uniform_int_distribution<unsigned int> u(0, 10000);

	for (long int i = 0; i < dataline; i++)
	{
		for (int j = 0; j < key_num; j++)
		{
			fprintf(fp, "%u\t", u(rng));
		}
		fprintf(fp, "%d\n", i + 1);
	}

	fclose(fp);
}
