#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <omp.h>
using namespace std;

string X = "ACCCTAACCCTAACAACAACAACAACAACAACAACAAC";
string Y = "GGATCCGTCCGTCGTCGTCGTCGTCCGGC";
vector<vector<int>> table; // table of Dynamic Programming
set<string> setOfLCS1;      // Longest Common String set
set<string> setOfLCS2;      // Longest Common String set
/**
 * used for DP
 */
int max(int a, int b)
{
    return (a>b)? a:b;
}

/**
 * up-side-down the input string(because output will be reversed from the table)
 */
string Reverse(string str)
{
    int low = 0;
    int high = str.length() - 1;
    while (low < high)
    {
        char temp = str[low];
        str[low] = str[high];
        str[high] = temp;
        ++low;
        --high;
    }
    return str;
}

/**
 * build table and get the length of LCS
 *
 * at first X0 == Y0 == 0
 * if Xi == Yj table[i][j] = max(table[Xi-1, Yj], table[X, Yj-1]);
 * if Xi == Yi table[i][j] = table[i-1][j-1] + 1;
 * So then we can get a 2D array, the length is the final value of the array
 */
int DPTableInit(int m, int n)
{
    // first row and col are set to 0
    table = vector<vector<int>>(m+1,vector<int>(n+1));

    for(int i=0; i<m+1; ++i)
    {
        for(int j=0; j<n+1; ++j)
        {
            // first row and second row is zero
            if (i == 0 || j == 0) {
                table[i][j] = 0;
            }
            else if(X[i-1] == Y[j-1]) {
                table[i][j] = table[i - 1][j - 1] + 1;
            }
            else {
                table[i][j] = max(table[i - 1][j], table[i][j - 1]);
            }
        }
    }
    return table[m][n];
}

/**
 *
 */
void PrintLCS(int i, int j)
{
    if (i == 0 || j == 0)
        return;
#pragma parallel
    {
#pragma omp single
        {
            if (table[i][j] == table[i - 1][j]) {
#pragma omp task
                PrintLCS(i - 1, j);
            } else if (table[i][j] == table[i][j - 1]) {
#pragma omp task
                PrintLCS(i, j - 1);
            }else {
#pragma omp task
                PrintLCS(i - 1, j - 1);
                printf("%c ", X[i - 1]);
            }
        }
    }
}
/**
 * This function is used to trace all subsquence of two DNA
 *
 */
void ParallelTraceBackAll(int i, int j, int thread, string res_str) {

    while (i > 0 && j > 0) {
        if (X[i - 1] == Y[j - 1]) {
            res_str.push_back(X[i - 1]);
            i--;
            j--;
        } else {
            if (table[i - 1][j] > table[i][j - 1]) {
                i--;//search from right to left
            } else if (table[i - 1][j] < table[i][j - 1]) {
                j--;//search form bottom to up
            } else {//The values are the same,then this element is considered as the end of the 2D array to do another iteration
                omp_set_num_threads(1);
#pragma omp parallel
                {
#pragma omp single
                    {
#pragma omp task
                        ParallelTraceBackAll(i - 1, j, thread, res_str);
#pragma omp task
                        ParallelTraceBackAll(i, j - 1, thread, res_str);
                    }
                }
                return;
            }
        }
    }
    setOfLCS2.insert(Reverse(res_str));
}

void TraceBackAll(int i, int j, int thread, string res_str) {

    while (i > 0 && j > 0) {
        if (X[i - 1] == Y[j - 1]) {
            res_str.push_back(X[i - 1]);
            i--;
            j--;
        } else {
            if (table[i - 1][j] > table[i][j - 1]) {
                i--;//search from right to left
            } else if (table[i - 1][j] < table[i][j - 1]) {
                j--;//search form bottom to up
            } else {//The values are the same,then this element is considered as the end of the 2D array to do another iteration
                        TraceBackAll(i - 1, j, thread, res_str);
                        TraceBackAll(i, j - 1, thread, res_str);
                return;
            }
        }
    }
    setOfLCS1.insert(Reverse(res_str));
}

int main()
{
    int m = (int)X.length();
    int n = (int)Y.length();
    int length = DPTableInit(m, n);
    cout << "The length of LCS is " << length << endl;
    string str1;
    string str2;
    //PrintLCS(m, n);
    TraceBackAll(m, n, 8, str1);
    set<string>::iterator beg1 = setOfLCS1.begin();
    for( ; beg1!=setOfLCS1.end(); ++beg1)
        cout << *beg1 << endl;
    printf("-------------");
    ParallelTraceBackAll(m,n,8,str2);
    // 倒序输出
    set<string>::iterator beg2 = setOfLCS1.begin();
    for( ; beg2!=setOfLCS1.end(); ++beg2)
        cout << *beg2 << endl;
    return 0;
}