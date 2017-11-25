### Sudoku Solver
This is an sudoku-solver implementation coded in three days for GPU-class homework.

```bash
nvcc -arch sm_35 -rdc=true -o sudokusolver sudokusolver.cu
./sudokusolver inp.in
```

Which will print out the stdout and save the results in to the inp.sol file.
## Parallelizing Sudoku
is hard. The most popular cpu-solution is backtracking, which is built on backtracking and recursion. [This webpage](https://www.sudoku-solutions.com/) is really nice to improve your sudoku skills and it has `View Steps` functionality to look the logic behind the solution.

However since it is all about parallelizing I've focused on parallelizing and only implemented the basic logic. Basic logic is this:
1. For each cell(out of 81) if it is empty, find out the set of digits which are not used in the current row, column or cell-group. This set is the possibility set of each cell.
2. If the set is empty, this setting/board is invalid.(which may happen as a result of incorrect guessing)
3. If the set consists of single element, fill the cell with the only value you got.
4. If there are more then one digits in the set, then we don't do anything.

We keep repeat this process until we solve the sudoku or stop progressing. When there is no progress we schedule a fork:
1. Find the number of digits in the smallest set.
2. From the cell with smallest sets, pick the one with smaller id(id here is the row-based 1d matrix id)
3. Fork the cell by generating new boards for possible values of the cell. Each board gets another digit and they apply the simple logic again.

Yes the problem here is to how to fork. Gpu's have some recursion and dynamic allocation capability, but it is always better to allocate at the beginning and thats what I do.

## Optimizations and Results
Current:
- Static allocation with 50000 blocks(observed to be enough for all hard examples)
- Bit masks to reduce storaga.

Future:
- Share bit mask generation task better within block.

<2s kernel time for 95 hard sudoku.

## What cuda-sudoku-solver does.
The `controller` kernel is the main Kernel which calls  `fillSudokuSafeAndFork` repeatedly until a solution is found.
The program has some default values like #blocks and #threads.
- `#threads`: 96=32*3 which is the smallest multiple of 32 which is bigger then 81. #todo We can do 81 here and remove if statements.
- `#blocks`: available solvers. Each block works on its own block.
- `arr_dev`: has #blocks many boards and one extra for the solution.
- `block_stat`:  has status for each block. If it is 0, block is idle/available. If it is 1, then it is active, working on a solution. If the last element of block_stat(block_stat[nBlocks]) is equal to 2, we have a solution ready on the last 81 element of `arr_dev`.

**fillSudokuSafeAndFork** is pretty long kernel, following steps are done in order:
1. block 0 checks for errors, it stops the process if there is no active block. This shouldn't happen.
2. Each active block calculates row,column,cell-group binary masks(9bit). Using binary masks reduces the shared memory requirement.
3. Each thread is matched with a sudoku cell and each thread calculates its possibilities by OR'ing its corresponding masks. The result is the set of possible values(0's in the binary mask).
4. 2 and 3 repeated until no progress is made.
5. after the loop
  - if **done_flag**, then we copy the result to the result spot and set the `stats[nBlocks]` to 2.
  - if **error_flag**, current block is wrong, so we spot and set the stat to 0, so the block can be rescheduled.
  - if no **progress_flag**, then we need to fork.
    1. Using atomic instructions choose a cell with multiple possibilities. After this point only the corresponding thread performs.
    2. First possibility stays with the block. For the remaining digits make a search to find an available block and copy the current block to the new_blocks global storage. Now in the next iteration the new blocks are going to work on different possibilities.
