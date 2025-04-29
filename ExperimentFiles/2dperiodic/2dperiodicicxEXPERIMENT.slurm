#!/bin/bash
#SBATCH --job-name=2dperiodicicx        # Job name
#SBATCH --output=icx_job_output.txt         # Output file
#SBATCH --error=icx_job_error.txt           # Error file
#SBATCH --ntasks=1                      # Number of tasks (1 task per node)
#SBATCH --cpus-per-task=80              # Number of CPUs per task
#SBATCH --time=14:00:00                 # Max run time (hh:mm:ss)
#SBATCH --mem=128GB                      # Memory per node
#SBATCH --partition=icx                 # Partition to run the job on
#SBATCH --nodes=1                       # Number of nodes (1 node for this job)

# Load modules
module load intel/24.0
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80

# Set the working directory (where plutinput.c is located)
cd /home1/10457/michaelsbu/pluto/2dperiodic

sed -i "s/#define N.*/#define N 8000L/" plutoinputicx.c

# Define the different values for TIME
t_values=(3 30 300 3000 30000 300000)

#echo " " > outputicx.txt
cat tile.sizes >> outputicx.txt


for t in "${t_values[@]}"; do
    echo "Current value = $t" >> outputicx.txt
    sed -i "s/#define T.*/#define T ${t}L/" plutoinputicx.c

    # Array to store results for this t
    declare -a results

    for ((i=1; i<=10; i++)); do
        ../polycc plutoinputicx.c --tile --parallel
        icx -xhost -O3 -qopenmp -ansi-alias -ipo -D TIME -mkl plutoinputicx.pluto.c -o icxexe
        output=$(./icxexe)
        echo "$output" >> outputicx.txt

        # Extract runtime from the output
        runtime=$(echo "$output" | grep "Time taken" | awk -F'|' '{print $3}' | awk '{print $4}' | sed 's/s//')
        results+=("$runtime")
    done

done
