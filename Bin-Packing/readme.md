# First Fit Data Generator with Packed Bin Output

This C++ program implements the First Fit algorithm for the bin packing problem and generates a data file. Each line in the output file contains:
1.  A list of randomly generated item sizes (input) on the left side of the " | " separator.
2.  A string representation of the bins and their contents after applying the First Fit algorithm to these items, on the right side of the " | " separator.

**Output Format Details (Right Side):**
The representation of packed bins is a string where:
* Items within each bin are comma-separated (e.g., `item1,item2,item3`).
* Bins themselves are separated by a semicolon (`;`).
* Example: If First Fit results in Bin 1 containing items `23,61` and Bin 2 containing items `40,11,29`, the right side string would be `23,61;40,11,29`.

The items are processed by the First Fit algorithm in the order they were randomly generated.

## How to Compile and Run

1.  **Save the Code**: Save the C++ code as `first_fit_generator.cpp` (or any other `.cpp` filename).
2.  **Compile**: Use a C++ compiler (like g++) to compile the code. You'll need C++11 or newer for features like `<random>`.
    ```bash
    g++ first_fit_generator.cpp -o first_fit_generator -std=c++11
    ```
3.  **Run**: Execute the compiled program.
    ```bash
    ./first_fit_generator
    ```
    This will create a file (default: `first_fit_packed_data.txt`) in the same directory.

## Modifying Data Generation Parameters

To change the characteristics of the generated data, you can modify the constant variables defined at the beginning of the `main()` function in the `first_fit_generator.cpp` file.

```cpp
// Inside main() function:

    // --- Configuration Parameters for Data Generation ---
    const int NUM_LINES = 3000;           // Number of lines to generate in the output file
    const int ITEMS_PER_LINE = 2000;      // Number of items to generate for each line
                                          // Note: A large number of items can make the packed bin string very long.
                                          // You might adjust this for readability or specific needs.
    const int MAX_ITEM_SIZE = 100;        // Maximum size of a randomly generated item
    const int MIN_ITEM_SIZE = 1;          // Minimum size of a randomly generated item
    const int BIN_CAPACITY = 100;         // Capacity of each bin (for the First Fit algorithm)
    const std::string OUTPUT_FILENAME = "first_fit_packed_data.txt"; // Name of the output file
    // --- End of Configuration Parameters ---