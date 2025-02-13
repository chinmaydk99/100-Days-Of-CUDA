
#include <iostream>
#include <vector>

using namespace std;

void prefixSum(const vector<int>&arr)
{
    vector<int> prefix(arr.size());
    prefix[0] = arr[0];
    
    for(size_t i = 1; i < arr.size(); i++)
    {
        prefix[i] = prefix[i-1] + arr[i];
    }
    
    for(const auto &num:prefix)
    {
        cout<<num << " ";
    }
}

int main()
{
    vector<int> input = {1,2,3,4,5};
    prefixSum(input);

    return 0;
}