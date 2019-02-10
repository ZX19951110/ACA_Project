/**
 * We use two algorithms to match substring of a long text and record the index
 * The first algorithm is Brute Force which time complexity is O(m * n), and we use "omp parallel for" to optimize the loop with static schedule, the chunk of the optimize is depend on the threads
 * The second algorithm is Rabin-Karp which time complexity is O(n) in the best situation, and O((m-n+1)n) in the worst situation, We try to use the "omp parallel task" to optimize the algorithm
*/
#include <iostream>
#include <string.h>
#include <vector>
#include <set>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

set<int> BruteForce1;
set<int> BruteForce2;
set<int> RabinKarp1;
set<int> RabinKarp2;
/**
 * original BruteForce algorithm without OpenMP
 *
 */
double BruteForce(){
    double start = omp_get_wtime();
    //prepare the file to be matched into memory
    FILE*fp1;
    fp1=fopen("bbe.txt","r");
    fseek(fp1,0L,SEEK_END);
    int size1=ftell(fp1);
    char* str =(char *)malloc(size1);
    fseek(fp1,0L,SEEK_SET);
    fread(str,size1,1,fp1);
    fclose(fp1);
    //prepare the matching string to be load into memory
    FILE*fp2;
    fp2=fopen("match.txt","r");
    fseek(fp2,0L,SEEK_END);
    int size2=ftell(fp2);
    char* match =(char *)malloc(size2);
    fseek(fp2,0L,SEEK_SET);
    fread(match,size2,1,fp2);
    fclose(fp2);

    for (int i = 0; i <= size1-size2; i++){
        int status = 1; //乐观锁
        for(int j = 0; j < size2; j++){
            if(str[i+j] != match[j]){
                status = 0;
                //break; here should be break, but if break, the time complexity will be O(m-Tmiss+Tmiss*n) rather than O(m * n)
            }
        }
        if(status == 1){
            BruteForce1.insert(i);
        }
    }
    double end = omp_get_wtime();
    set<int>::iterator i = BruteForce1.begin();
    for(; i!=BruteForce1.end(); ++i) {
        printf("Index found is %d\n", *i);
    }
    double time_spent = end-start;
    printf("BruteForce algorithm without omp spend: %fs\n",time_spent);
    return time_spent;
}
/**
 *BruteForce algorithm with OpenMP use "parallel omp for" to optimize, In the iteration, we use static schedule and each chunk size depend on the number of threads
 * @param num_threads
 *
 */
double BruteForceParallel(int num_threads){
    omp_set_num_threads(num_threads);
    char* str;
    char* match;
    int size1, size2;
    int chunk;
    double start = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp sections
        {
#pragma omp section
            {
                //prepare the file to be matched into memory
                FILE*fp1;
                fp1=fopen("bbe.txt","r");
                fseek(fp1,0L,SEEK_END);
                size1=ftell(fp1);
                str =(char *)malloc(size1);
                fseek(fp1,0L,SEEK_SET);
                fread(str,size1,1,fp1);
                fclose(fp1);
        }
#pragma omp section
            {
                //prepare the matching string to be load into memory
                FILE*fp2;
                fp2=fopen("match.txt","r");
                fseek(fp2,0L,SEEK_END);
                size2=ftell(fp2);
                match =(char *)malloc(size2);
                fseek(fp2,0L,SEEK_SET);
                fread(match,size2,1,fp2);
                fclose(fp2);
            }
    }
#pragma omp single
        {
            chunk = size1 / num_threads; // to allocate the size of chunk of static schedule
        }


#pragma omp for schedule(static, chunk)
        for (int i = 0; i <= size1 - size2; i++) {
            int status = 1; //乐观锁
            for (int j = 0; j < size2; j++) {
                if (str[i + j] != match[j]) {
                    status = 0;
                }
            }
            if (status == 1) {
                BruteForce2.insert(i);
            }
        }
    }
    double end = omp_get_wtime();
    set<int>::iterator i = BruteForce2.begin();
    for(; i!=BruteForce2.end(); ++i) {
        printf("Index found is %d\n", *i);
    }
    double time_spent = end-start;
    printf("BruteForce algorithm with omp for spend: %fs\n",time_spent);
    return time_spent;
}
/**
 * Original RabinKarp algorithm, we use rolling hash to compute the digist of each result of the iteration, it can reduce the time complexity
 *
 */
 double RabinKarp(){
    double start = omp_get_wtime();
    //prepare the file to be matched into memory
    FILE*fp1;
    fp1=fopen("bbe.txt","r");
    fseek(fp1,0L,SEEK_END);
    int size1=ftell(fp1);
    char* str =(char *)malloc(size1);
    fseek(fp1,0L,SEEK_SET);
    fread(str,size1,1,fp1);
    fclose(fp1);
    //prepare the matching string to be load into memory
    FILE*fp2;
    fp2=fopen("match.txt","r");
    fseek(fp2,0L,SEEK_END);
    int size2=ftell(fp2);
    char* match =(char *)malloc(size2);
    fseek(fp2,0L,SEEK_SET);
    fread(match,size2,1,fp2);
    fclose(fp2);
    int mod = 0x7fffffff;//the biggest primary number
    const int BASE = 256;//a char must less than 256
    int h = 1;//for rolling
    for (int i = 0; i < size2 - 1; i++)
        h = (h * BASE) % mod;
    int match_hash = 0;
    int str_hash = 0;
    for (int i = 0; i < size2; i++) {
        match_hash = (BASE * match_hash + match[i]) % mod;//the hash of the match string
        str_hash = (BASE * str_hash + str[i]) % mod;//initialize the first hash of the rolling hash
    }
    for (int i = 0; i <= size1-size2; i++) {
        if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
            RabinKarp1.insert(i);
        }
        else {
            str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
        }
    }
    double end = omp_get_wtime();
    set<int>::iterator i = RabinKarp1.begin();
    for(; i!=RabinKarp1.end(); ++i) {
        printf("Index found is %d\n", *i);
    }
    double time_spent = end-start;
    printf("RabinKarp algorithm without omp spend: %fs\n",time_spent);
    return time_spent;
 }
/**
 * this fuction use "section" to optimize the code, although in rolling hash the next hash is depend on this iteration, but if we divide the whole text into chunks and the number is the thread number, we can unroll the iteration and use parallel mode to execute with compute every initial value of first rolling hash value of the chunks.
 * @param num_threads
 * @return
 */
 double RabinKarpParallel(int num_threads){
     omp_set_num_threads(num_threads);
     char* str;
     char* match;
     int size1, size2;
     int mod = 0x7fffffff;//the biggest primary number
     const int BASE = 256;//a char must less than 256
     int h = 1;//for rolling
     int match_hash = 0;
     int str_hash = 0;
     double start = omp_get_wtime();
     int chunk;
#pragma omp parallel
     {
#pragma omp sections
         {
#pragma omp section
             {
                 //prepare the file to be matched into memory
                 FILE*fp1;
                 fp1=fopen("bbe.txt","r");
                 fseek(fp1,0L,SEEK_END);
                 size1=ftell(fp1);
                 str =(char *)malloc(size1);
                 fseek(fp1,0L,SEEK_SET);
                 fread(str,size1,1,fp1);
                 fclose(fp1);
             }
#pragma omp section
             {
                 //prepare the matching string to be load into memory
                 FILE*fp2;
                 fp2=fopen("match.txt","r");
                 fseek(fp2,0L,SEEK_END);
                 size2=ftell(fp2);
                 match =(char *)malloc(size2);
                 fseek(fp2,0L,SEEK_SET);
                 fread(match,size2,1,fp2);
                 fclose(fp2);
             }
         }
#pragma omp single
         {
             chunk = size1 / num_threads;
             for (int i = 0; i < size2; i++) {
                 match_hash = (BASE * match_hash + match[i]) % mod;
             }
         }

//#pragma omp barrier

#pragma omp sections nowait private(str_hash)
         {
#pragma omp section
             {
                 int left = chunk * 0;
                 int right = chunk * 1 - size2;
                 for (int i = left; i < left + size2 - 1; i++)
                     h = (h * BASE) % mod;
                 for (int i = left; i < left + size2; i++) {
                     str_hash = (BASE * str_hash + str[i]) % mod;
                 }
                 for (int i = left; i <= right; i++) {
                     if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                         RabinKarp2.insert(i);
                     }
                     else {
                         str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                     }
                 }
         }
#pragma omp section
             {
                 int left = chunk * 1;
                 int right = chunk * 2 - size2;
                 for (int i = left; i < left + size2 - 1; i++)
                     h = (h * BASE) % mod;
                 for (int i = left; i < left + size2; i++) {
                     str_hash = (BASE * str_hash + str[i]) % mod;
                 }
                 for (int i = left; i <= right; i++) {
                     if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                         RabinKarp2.insert(i);
                     }
                     else {
                         str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                     }
                 }
             }
#pragma omp section
             {
                 int left = chunk * 2;
                 int right = chunk * 3 - size2;
                 for (int i = left; i < left + size2 - 1; i++)
                     h = (h * BASE) % mod;
                 for (int i = left; i < left + size2; i++) {
                     str_hash = (BASE * str_hash + str[i]) % mod;
                 }
                 for (int i = left; i <= right; i++) {
                     if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                         RabinKarp2.insert(i);
                     }
                     else {
                         str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                     }
                 }
             }
#pragma omp section
            {
                int left = chunk * 3;
                int right = chunk * 4 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 4;
                int right = chunk * 5 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 5;
                int right = chunk * 6 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 6;
                int right = chunk * 7 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 7;
                int right = chunk * 8 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 8;
                int right = chunk * 9 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 9;
                int right = chunk * 10 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 10;
                int right = chunk * 11 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 11;
                int right = chunk * 12 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 12;
                int right = chunk * 13 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 13;
                int right = chunk * 14 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
            {
                int left = chunk * 14;
                int right = chunk * 15 - size2;
                for (int i = left; i < left + size2 - 1; i++)
                    h = (h * BASE) % mod;
                for (int i = left; i < left + size2; i++) {
                    str_hash = (BASE * str_hash + str[i]) % mod;
                }
                for (int i = left; i <= right; i++) {
                    if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                        RabinKarp2.insert(i);
                    }
                    else {
                        str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                    }
                }
            }
#pragma omp section
             {
                 int left = chunk * 15;
                 int right = size1 - size2;//for last chunk
                 for (int i = left; i < left + size2 - 1; i++)
                     h = (h * BASE) % mod;
                 for (int i = left; i < left + size2; i++) {
                     str_hash = (BASE * str_hash + str[i]) % mod;
                 }
                 for (int i = left; i <= right; i++) {
                     if (match_hash == str_hash && memcmp(match, str + i, size2) == 0) {
                         RabinKarp2.insert(i);
                     }
                     else {
                         str_hash = (BASE*(str_hash - h*str[i]) + str[i+size2]) % mod;
                     }
                 }
             }
         }
     }
     double end = omp_get_wtime();
     set<int>::iterator i = RabinKarp2.begin();
     for(; i!=RabinKarp2.end(); ++i) {
         printf("Index found is %d\n", *i);
     }
     double time_spent = end-start;
     printf("RabinKarp algorithm with omp task spend: %fs\n",time_spent);
     return time_spent;
 }
int main(){
    int num_threads = 16;
    double time1 = BruteForce();
    double time2 = BruteForceParallel(num_threads);
    printf("the speedup of BruteForce with %d threads is %f\n", num_threads,time1/time2);
    cout<<"-----------------------\n"<<endl;
    double time3 = RabinKarp();
    double time4 = RabinKarpParallel(num_threads);
    printf("the speedup of RabinKarp with %d threads is %f\n", num_threads,time3/time4);
}
//thank you for your reading :D