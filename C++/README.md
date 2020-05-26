# Advanced Concepts of C++

#### Multithreading

Multithreading is a specialized way of multitasking. This behavior allows the user to perform multiple tasks simultaneously. For any action taken by a user on the system, there must be a process to complete that action (as asked by a user). Every process must have at least one thread associated with it. The OS built in every system is responsible to allocate the processing time for every thread. 

std::thread is the thread class that represents a single thread in C++. To start a thread we simply need to create a new thread object and pass the executing code to be called (i.e, a callable object) into the constructor of the object. Once the object is created a new thread is launched which will execute the code specified in callable.

A callable can be either of the three
- A function pointer 
- A function object (Functor)
- A lambda expression


