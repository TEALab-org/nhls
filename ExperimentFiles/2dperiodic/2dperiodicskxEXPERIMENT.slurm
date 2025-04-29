#!/bin/bash
#SBATCH --job-name=2dperiodicskx        # Job name
#SBATCH --output=skx_job_output.txt         # Output file
#SBATCH --error=skx_job_error.txt           # Error file
#SBATCH --ntasks=1                      # Number of tasks (1 task per node)
#SBATCH --cpus-per-task=48              # Number of CPUs per task
#SBATCH --time=010:00:00                 # Max run time (hh:mm:ss)
#SBATCH --mem=128GB                      # Memory per node
#SBATCH --partition=skx                 # Partition to run the job on
#SBATCH --nodes=1                       # Number of nodes (1 node for this job)

# Load modules
module load intel/24.0
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48

# Set the working directory (where plutinput.c is located)
cd /home1/10457/michaelsbu/pluto/2dperiodic

sed -i "s/#define N.*/#define N 8000L/" plutoinputskx.c


# Define the different values for TIME
t_values=(1 10 100 1000 10000 100000 1000000)

echo " " > outputskx.txt
cat tile.sizes >> outputskx.txt

for t in "${t_values[@]}"; do
    echo "Current value = $t" >> outputskx.txt
    sed -i "s/#define T.*/#define T ${t}L/" plutoinputskx.c

    # Array to store results for this t
    declare -a results

    for ((i=1; i<=10; i++)); do
        ../polycc plutoinputskx.c --tile --parallel
        icx -xhost -O3 -qopenmp -ansi-alias -ipo -D TIME -mkl plutoinputskx.pluto.c -o skxexe
        output=$(./skxexe)
        echo "$output" >> outputskx.txt

        # Extract runtime from the output
        runtime=$(echo "$output" | grep "Time taken" | awk -F'|' '{print $3}' | awk '{print $4}' | sed 's/s//')
        results+=("$runtime")
    done

done
