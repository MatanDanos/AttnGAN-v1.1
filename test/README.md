# Unit Tests

The unit tests should allow us to quickly test our entire model pipeline quickly before starting the actual training on the entire dataset.  
In order for this to work correctly, every component of the source code should have a unit-test file that tests its functionality:

* Simple and short test should suffice.  
* The main objective is to complete an entire pipeline of the code before the main training, with high code coverage.  

## Running Tests

To run all tests, either use your IDE or type the following commmands in the terminal:

* To run all tests: `python -m unittest discover ../`
  * This command should be activated from the `./src` directory
  * Adding the `-v` flag helps for more verbose respones.
* To run a specific test file: `python -m unittest test.test_filename`
* To run all tests in a specific subdirectory: `python -m unittest test.test_subdir`
* To run a specific TestCase in a specific test file: `python -m unittest test.test_filename.TestCase`
* To run a specific test method: `python -m unittest test.test_filename.TestCase.test_method`

## Example_Dataset Directory

The Example_Dataset folder will contain only a few image and text examples from the CUB-200-2011 dataset.  
This small dataset will help us peform quick unit test.  
The directory should be uploaded to github (possible due to its relative small size).  