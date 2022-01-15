# DetailedGlider        
 
##### Operational Sequence
1. Ocean Glider Python Spec\
    This python implementation of the project is meant to serve as the definitive specification for the functionality of the embedded code module. Once the code in this spec is documented, verified to be correct in its operation, and verified to be feature complete proceed to step 2. 
2. Python to C Blueprint\
    In this step the specification from step 1 must be translated into a version of Python more analogous to idiomatic C code. The purpose of this step is to remove the training wheels afforded by Numpy and to shine light on any functionality that may require extra research for a correct C implementation. 
3. C implementation\
    The C implementation should follow the blueprint laid out by Step 2 as closely as possible. The functionality and structure of the code should remain unchanged. Only the implementation details will differ.
