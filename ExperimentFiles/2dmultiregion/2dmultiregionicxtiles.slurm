#!/bin/bash
#SBATCH --job-name=2dmultiregionicx         # Job name
#SBATCH --output=icx_job_output.txt         # Output file
#SBATCH --error=icx_job_error.txt           # Error file
#SBATCH --ntasks=1                      # Number of tasks (1 task per node)
#SBATCH --cpus-per-task=80              # Number of CPUs per task
#SBATCH --time=03:00:00                # Max run time (hh:mm:ss)
#SBATCH --mem=128GB                     # Memory per node
#SBATCH --partition=icx                 # Partition to run the job on
#SBATCH --nodes=1                       # Number of nodes (1 node for this job)

# Load modules
module load intel/24.0
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80

# Set the working directory (where plutoinput.c is located)
cd /home1/10457/michaelsbu/pluto/2dmultiregion

tile_sizes1=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
tile_sizes2=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
tile_sizes3=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)

# Define the different values for TIME
time_values=(1000)

echo " " > outputicx.txt
for time in "${time_values[@]}"; do
  echo "Current TIME = $time" >> outputicx.txt

  # Update the TIME value in the plutoinputicx.c file
  sed -i "s/#define T.*/#define T ${time}L/" plutoinputicx.c

  # Loop through all combinations of tile sizes for T and N
  for t1 in "${tile_sizes1[@]}"; do
    for t2 in "${tile_sizes2[@]}"; do
      #for t3 in "${tile_sizes2[@]}"; do
      echo $t1 > tileicx/tile.sizes  # Overwrite tile.sizes with the first tile size
      echo $t1 >> tileicx/tile.sizes
      echo $t2 >> tileicx/tile.sizes
      echo "Current tile sizes: " >> outputicx.txt
      cat tileicx/tile.sizes >> outputicx.txt
      cd tileicx
      # Recompile plutoinput.c with polycc
      ../../polycc ../plutoinputicx.c --tile --parallel -o ../plutoinputicx.pluto.c
      cd ..
      # Compile the tiled code using icx
      icx -xhost -O3 -qopenmp -ansi-alias -ipo -D TIME -mkl plutoinputicx.pluto.c -o icxexe

      # Run the compiled executable and write the output to outputicx.txt
      ./icxexe >> outputicx.txt
      #done
    done
  done
done

./getmin.sh outputicx.txt
