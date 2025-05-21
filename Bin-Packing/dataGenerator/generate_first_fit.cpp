#include <iostream>
#include <vector>
#include <string>
#include <sstream>     // For string streams
#include <fstream>     // For file operations
#include <algorithm>   // For sorting (though not used for the final output format anymore)
#include <random>      // For random number generation
#include <numeric>     // For std::accumulate (if calculating bin fullness, or other uses)

// First Fit bin packing algorithm implementation
// Parameters:
//   items - A vector containing the sizes of all items
//   binCapacity - The capacity of each bin
// Returns:
//   A 2D vector where each inner vector represents a bin and its contained items
std::vector<std::vector<int>> firstFit(const std::vector<int>& items, int binCapacity) {
    std::vector<std::vector<int>> bins; // Stores all bins and the items within them
    std::vector<int> binRemainingCapacity; // Stores the remaining capacity of each bin

    if (binCapacity <= 0) {
        // If bin capacity is invalid, do nothing
        return bins;
    }

    for (int item : items) {
        if (item <= 0 || item > binCapacity) {
            // Skip invalid items or items too large to fit in any single bin
            // Alternatively, could throw an error or handle specially based on requirements
            continue;
        }

        bool placed = false;
        // Try to place the item into an existing bin
        for (size_t i = 0; i < bins.size(); ++i) {
            if (item <= binRemainingCapacity[i]) {
                bins[i].push_back(item);
                binRemainingCapacity[i] -= item;
                placed = true;
                break;
            }
        }

        // If no suitable existing bin is found, open a new bin
        if (!placed) {
            bins.push_back({item}); // Create a new bin and add the current item
            binRemainingCapacity.push_back(binCapacity - item);
        }
    }
    return bins;
}

// Helper function: Joins vector elements into a string with a given delimiter
std::string joinVector(const std::vector<int>& vec, char delimiter) {
    std::stringstream ss;
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec[i];
        if (i < vec.size() - 1) {
            ss << delimiter;
        }
    }
    return ss.str();
}

// Helper function: Formats the packed bins into the specified string representation
// e.g., {{23,61},{62,11}} -> "23,61;62,11"
std::string formatPackedBinsToString(const std::vector<std::vector<int>>& packedBins) {
    std::stringstream ss;
    for (size_t i = 0; i < packedBins.size(); ++i) {
        ss << joinVector(packedBins[i], ','); // Items within a bin are comma-separated
        if (i < packedBins.size() - 1) {
            ss << ';'; // Bins are semicolon-separated
        }
    }
    return ss.str();
}


int main() {
    // --- Configuration Parameters for Data Generation ---
    const int NUM_LINES = 3000;           // Number of lines to generate in the output file
    const int ITEMS_PER_LINE = 20;        // Number of items to generate for each line (Reduced for better readability of bin output)
    const int MAX_ITEM_SIZE = 100;        // Maximum size of a randomly generated item
    const int MIN_ITEM_SIZE = 1;          // Minimum size of a randomly generated item
    const int BIN_CAPACITY = 100;         // Capacity of each bin (for the First Fit algorithm)
    const std::string OUTPUT_FILENAME = "first_fit_packed_data.txt"; // Name of the output file (updated)
    // --- End of Configuration Parameters ---

    // Setup random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(MIN_ITEM_SIZE, MAX_ITEM_SIZE);

    std::ofstream outfile(OUTPUT_FILENAME);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file " << OUTPUT_FILENAME << std::endl;
        return 1;
    }

    std::cout << "Generating data file: " << OUTPUT_FILENAME << "..." << std::endl;

    for (int i = 0; i < NUM_LINES; ++i) {
        std::vector<int> currentItems;
        currentItems.reserve(ITEMS_PER_LINE);
        for (int j = 0; j < ITEMS_PER_LINE; ++j) {
            currentItems.push_back(distrib(gen));
        }

        // Left side: Original list of items (comma-separated)
        std::string inputStr = joinVector(currentItems, ',');

        // Apply First Fit algorithm
        std::vector<std::vector<int>> packedBins = firstFit(currentItems, BIN_CAPACITY);
        
        // Right side: Formatted string of bin contents
        std::string ffOutputStr = formatPackedBinsToString(packedBins);

        // Write to file
        outfile << inputStr << " | " << ffOutputStr << std::endl;

        if ((i + 1) % 300 == 0) { // Provide progress update every 300 lines
            std::cout << "Generated " << (i + 1) << "/" << NUM_LINES << " lines..." << std::endl;
        }
    }

    outfile.close();
    std::cout << "File " << OUTPUT_FILENAME << " generated successfully." << std::endl;
    std::cout << "File format per line:" << std::endl;
    std::cout << "[Original comma-separated item sizes] | [FF Bin1_item1,item2;Bin2_item1,item2,...]" << std::endl;

    // Example of how the output from firstFit can be interpreted (for console, not file)
    // std::cout << "\n--- Example of First Fit detailed output (for one instance) ---" << std::endl;
    // if (NUM_LINES > 0 && ITEMS_PER_LINE > 0) { // Quick check if any items were generated
    //     std::vector<int> exampleItems;
    //     for(int k=0; k<5; ++k) exampleItems.push_back(distrib(gen)); // a small example
    //     std::cout << "Example Items: " << joinVector(exampleItems, ',') << std::endl;
    //     std::cout << "Bin Capacity: " << BIN_CAPACITY << std::endl;
    //     std::vector<std::vector<int>> examplePackedBins = firstFit(exampleItems, BIN_CAPACITY);
    //     std::cout << "Number of bins used by FF: " << examplePackedBins.size() << std::endl;
    //     std::cout << "Formatted FF Output String: " << formatPackedBinsToString(examplePackedBins) << std::endl;
    //     for (size_t k = 0; k < examplePackedBins.size(); ++k) {
    //         int binTotal = 0;
    //         for (int item_in_bin : examplePackedBins[k]) {
    //             binTotal += item_in_bin;
    //         }
    //         std::cout << "Bin " << k + 1 << " (Total: " << binTotal << ", Remaining: "
    //                   << (BIN_CAPACITY - binTotal)
    //                   << "): " << joinVector(examplePackedBins[k], ',') << std::endl;
    //     }
    // }
    // std::cout << "--------------------------------------------------------------" << std::endl;


    return 0;
}