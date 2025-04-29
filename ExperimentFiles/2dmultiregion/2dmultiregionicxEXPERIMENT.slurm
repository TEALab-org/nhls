#!/bin/bash
#SBATCH --job-name=2dmultiregionicx        # Job name
#SBATCH --output=icx_job_output.txt         # Output file
#SBATCH --error=icx_job_error.txt           # Error file
#SBATCH --ntasks=1                      # Number of tasks (1 task per node)
#SBATCH --cpus-per-task=80              # Number of CPUs per task
#SBATCH --time=010:00:00                 # Max run time (hh:mm:ss)
#SBATCH --mem=128GB                      # Memory per node
#SBATCH --partition=icx                 # Partition to run the job on
#SBATCH --nodes=1                       # Number of nodes (1 node for this job)

# Load modules
module load intel/24.0
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80

# Set the working directory (where plutinput.c is located)
cd /home1/10457/michaelsbu/pluto/2dmultiregion

# Generate t_values array
t_values=()
for i in {27..30}; do
    value=$((2**i))
    sqrt=$(echo "scale=0; sqrt($value)/1" | bc)
    t_values+=("$sqrt")
done



# Main script
echo " " > outputicx.txt
cat tile.sizes >> outputicx.txt

for t in "${t_values[@]}"; do
    echo "Current value = $t" >> outputicx.txt

    # Update the #define N and #define T lines in plutoinputicx.c
    sed -i "s/#define N.*/#define N ${t}L/" plutoinputicx.c
    sed -i "s/#define T.*/#define T ${t}L/" plutoinputicx.c

    # Array to store results for this t
    declare -a results

    for ((i=1; i<=10; i++)); do
        # Run polycc and compile the code
        ../polycc plutoinputicx.c --tile --parallel
        icx -xhost -O3 -qopenmp -ansi-alias -ipo -D TIME -mkl plutoinputicx.pluto.c -o icxexe

        # Run the executable and capture the output
        output=$(./icxexe)
        echo "$output" >> outputicx.txt

        # Extract runtime from the output
        runtime=$(echo "$output" | grep "Time taken" | awk -F'|' '{print $3}' | awk '{print $4}' | sed 's/s//')
        results+=("$runtime")
    done
done

