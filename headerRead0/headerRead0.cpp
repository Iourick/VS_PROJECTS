// headerRead0.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>


#include <fstream>
#include <string>

int main() {
    std::string file_name = "D://weizmann//RAW_DATA//blc20_guppi_57991_49905_DIAG_FRB121102_0011.0007.raw";  // Replace with the name of your ASCII file
    int num_characters = 160;  // Replace with the number of characters you want to read

    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cout << "Unable to open file." << std::endl;
        return 1;
    }

    // Read the first N characters into a string
    std::string content;
    content.resize(num_characters);  // Resize the string to hold N characters

    file.read(&content[0], num_characters);  // Read N characters from the file

    file.close();  // Close the file

    std::cout << "Read content: " << content << std::endl;  // Output the read content

    return 0;
}
