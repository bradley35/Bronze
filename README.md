# Bronze

The latest Macs and iOS devices have surprisingly powerful GPUs built right in.
Bronze is my attempt to use Metal (Apple's GPU shader language based on C++) to parallelize and execute some common matrix and sorting operations.

The included matrix operations work on both iOS and Mac devices.

Sorting works on Mac but currently makes use of simd groups and will therefore throw an error on all but the latest iOS devices. 
