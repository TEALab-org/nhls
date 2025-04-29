#!/bin/bash
#SBATCH --job-name=1dmultiregionskx        # Job name
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
cd /home1/10457/michaelsbu/pluto/1dmultiregionSKX

# Generate t_values array
t_values=(100000 1000000)
#for i in {26..26}; do
#    value=$((2**i))
#    sqrt=$(echo "scale=0; $value/1" | bc)
#    t_values+=("$sqrt")
#done



# Main script
echo " " > outputskx.txt
cat tile.sizes >> outputskx.txt

for t in "${t_values[@]}"; do
    echo "Current value = $t" >> outputskx.txt

    # Update the #define N and #define T lines in plutoinputskx.c
    sed -i "s/#define N.*/#define N 1600000L/" plutoinputskx.c
    sed -i "s/#define T.*/#define T ${t}L/" plutoinputskx.c

    # Array to store results for this t
    declare -a results

    for ((i=1; i<=10; i++)); do
        # Run polycc and compile the code
        ../polycc plutoinputskx.c --tile --parallel
        icx -xhost -O3 -qopenmp -ansi-alias -ipo -D TIME -mkl plutoinputskx.pluto.c -o skxexe

        # Run the executable and capture the output
        output=$(./skxexe)
        echo "$output" >> outputskx.txt

        # Extract runtime from the output
        runtime=$(echo "$output" | grep "Time taken" | awk -F'|' '{print $3}' | awk '{print $4}' | sed 's/s//')
        results+=("$runtime")
    done
done
