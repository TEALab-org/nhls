#!/bin/bash
#SBATCH --job-name=2dperiodicskx        # Job name
#SBATCH --output=skx_job_output.txt         # Output file
#SBATCH --error=skx_job_error.txt           # Error file
#SBATCH --ntasks=1                      # Number of tasks (1 task per node)
#SBATCH --cpus-per-task=48              # Number of CPUs per task
#SBATCH --time=016:00:00                 # Max run time (hh:mm:ss)
#SBATCH --mem=128GB                      # Memory per node
#SBATCH --partition=skx                 # Partition to run the job on
#SBATCH --nodes=1                       # Number of nodes (1 node for this job)

# Load modules
module load intel/24.0
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48

# Set the working directory (where plutinput.c is located)
cd /home1/10457/michaelsbu/pluto/2dperiodic

tile_sizes1=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
tile_sizes2=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
tile_sizes3=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)

# Define the different values for TIME
time_values=(1000)

echo " " > outputskx.txt
for time in "${time_values[@]}"; do
  echo "Current TIME = $time" >> outputskx.txt

  # Update the TIME value in the plutoinputskx.c file
  sed -i "s/#define T.*/#define T ${time}L/" plutoinputskx.c

  # Loop through all combinations of tile sizes for T and N
  for t1 in "${tile_sizes1[@]}"; do
    for t2 in "${tile_sizes2[@]}"; do
      for t3 in "${tile_sizes2[@]}"; do
      echo $t1 > tileskx/tile.sizes  # Overwrite tileskx.sizes with the first tile size
      echo $t2 >> tileskx/tile.sizes
      echo $t3 >> tileskx/tile.sizes
      echo "Current tileskx sizes: " >> outputskx.txt
      cat tileskx/tile.sizes >> outputskx.txt

      cd tileskx

      # Recompile plutoinputicx.c with polycc
      ../../polycc ../plutoinputskx.c --tile --parallel -o ../plutoinputskx.pluto.c
      cd ..
      # Compile the tiled code using icx
      icx -xhost -O3 -qopenmp -ansi-alias -ipo -D TIME -mkl plutoinputskx.pluto.c -o skxexe

      # Run the compiled executable and write the output to outputskx.txt
      ./skxexe >> outputskx.txt
      done
    done
  done
done

./getmin.sh outputskx.txt
