****SECTION ONE ___ ABOUT THE NEXA COMPILER NOTE
# Nexa Programming Language

Nexa is a simple custom programming language built with Python and LLVM for educational and experimental purposes.  
It supports variables, arithmetic operations, printing, and basic string handling.

## Features
- Variable declaration and assignment
- Integer and string literals
- Arithmetic operations (`+`, `-`, `*`, `/`)
- Print statements
- Basic error handling

## Requirements
Before running Nexa, ensure you have the following installed:

- Python 3.8 or higher
- `llvmlite` Python package

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/adjeijrdev/Nexa_Compiler.git
   cd nexa

******SECTION TWO ___ Create and activate a virtual environment:
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

****SECTION THREE  ___Install dependencies
pip install -r requirements.txt

****SECTION FOUR __Write your Nexa code in a file with the .nx extension. Example:
let name = "Daniel"
let age = 25
print "My name is " + name
print "I am " + age

****SECTION FIVE __Run the compiler:
python compiler.py yourfile.nx


