#include "../vendor/acutest/include/acutest.h"

#define TEST_ARRAYS_EQUAL(arr1, arr2, size, type)           \
    do                                                      \
    {                                                       \
        bool arrays_equal = true;                           \
        for (size_t i = 0; i < size && arrays_equal; i++)   \
        {                                                   \
            if (arr1[i] != arr2[i])                         \
            {                                               \
                arrays_equal = false;                       \
                TEST_MSG("Arrays differ at index %zu:", i); \
                TEST_MSG("Expected: " #type, arr1[i]);      \
                TEST_MSG("Got: " #type, arr2[i]);           \
            }                                               \
        }                                                   \
        TEST_CHECK(arrays_equal);                           \
    } while (0)