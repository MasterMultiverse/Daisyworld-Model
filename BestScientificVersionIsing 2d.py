#Every temperature is simulated independently. This prevents a temperature sweep from carrying thermal memory from the previous point.
#Thermalization data is discarded. Measurements begin only after the lattice has had time to approach equilibrium.
#The order parameter is ⟨∣m∣⟩\langle |m| \rangle⟨∣m∣⟩. On a finite zero-field lattice, using the absolute value prevents positive and negative magnetized states from cancelling.
#Susceptibility and heat capacity use equilibrium fluctuations. These quantities should develop peaks near the finite-lattice transition region. The standard Metropolis rule accepts energy-lowering flips directly and accepts energy-increasing flips with the Boltzmann probability.
#A dense temperature grid is used near TcT_cTc​.
#The exact infinite-lattice magnetization is included for comparison.
#Results are reproducible because the random-number seed is fixed.
#All numerical results are exported to ising_phase_transition_results.csv.
#A 300-DPI figure is saved as ising_phase_transition_results.png.


import csv
import math
import time

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SCIENTIFIC 2D ISING MODEL
#
# Square lattice with:
#   J = 1
#   k_B = 1
#   external magnetic field h = 0
#   periodic boundary conditions
#
# Measured quantities:
#   <|m|>       average absolute magnetization per spin
#   <e>         average energy per spin
#   chi         magnetic susceptibility per spin
#   C_v         heat capacity per spin
#
# Algorithm:
#   Vectorized checkerboard Metropolis updates
# ============================================================


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

# Linear lattice size. Total number of spins is L * L.
L = 50

# Number of equilibration sweeps at each temperature.
# These sweeps are not included in the measurements.
THERMALIZATION_SWEEPS = 1000

# Number of sweeps used for measurements.
MEASUREMENT_SWEEPS = 2000

# Record one measurement after this many sweeps.
# This reduces correlation between consecutive measurements.
SAMPLE_INTERVAL = 5

# Random-number seed for reproducible results.
RANDOM_SEED = 12345

# Interaction strength and Boltzmann constant.
J = 1.0
KB = 1.0

# Exact critical temperature for the infinite square-lattice
# Ising model when J = 1 and k_B = 1.
TC_EXACT = 2.0 / math.log(1.0 + math.sqrt(2.0))

# Output filenames.
CSV_FILENAME = "ising_phase_transition_results.csv"
FIGURE_FILENAME = "ising_phase_transition_results.png"


# ============================================================
# TEMPERATURE GRID
# ============================================================

# Use fewer points away from Tc and more points near Tc.
# The denser central region gives a clearer view of the
# susceptibility and heat-capacity peaks.

temperatures_low = np.linspace(
    1.50,
    2.00,
    8,
    endpoint=False
)

temperatures_critical = np.linspace(
    2.00,
    2.55,
    24,
    endpoint=False
)

temperatures_high = np.linspace(
    2.55,
    3.50,
    14
)

temperatures = np.unique(
    np.concatenate(
        [
            temperatures_low,
            temperatures_critical,
            temperatures_high,
            np.array([TC_EXACT])
        ]
    )
)

temperatures.sort()


# ============================================================
# RANDOM-NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# CHECKERBOARD MASKS
# ============================================================

# The square lattice is split into two checkerboard sublattices.
# Spins on one sublattice do not directly interact with other
# spins on the same sublattice.
#
# Therefore, all spins on one color can be considered for
# updating simultaneously.

row_indices, column_indices = np.indices((L, L))

black_mask = (row_indices + column_indices) % 2 == 0
white_mask = ~black_mask


# ============================================================
# INITIALIZE THE LATTICE
# ============================================================

def initialize_lattice():
    """
    Create an L x L lattice with random spins +1 and -1.
    """

    return rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=(L, L)
    )


# ============================================================
# NEIGHBOR SUM
# ============================================================

def neighbor_sum(spins):
    """
    Calculate the sum of the four nearest-neighbor spins
    for every lattice site.

    np.roll implements periodic boundary conditions.
    """

    top = np.roll(spins, 1, axis=0)
    bottom = np.roll(spins, -1, axis=0)
    left = np.roll(spins, 1, axis=1)
    right = np.roll(spins, -1, axis=1)

    return top + bottom + left + right


# ============================================================
# CHECKERBOARD METROPOLIS UPDATE
# ============================================================

def update_sublattice(spins, temperature, mask):
    """
    Perform a Metropolis update on one checkerboard sublattice.

    A spin flip is accepted automatically if delta_E <= 0.

    If delta_E > 0, it is accepted with probability:

        exp(-delta_E / T)

    Units J = 1 and k_B = 1 are used.
    """

    neighbors = neighbor_sum(spins)

    delta_energy = (
        2.0
        * J
        * spins
        * neighbors
    )

    random_values = rng.random(spins.shape)

    accepted = (
        mask
        & (
            (delta_energy <= 0.0)
            | (
                random_values
                < np.exp(
                    -delta_energy
                    / (KB * temperature)
                )
            )
        )
    )

    spins[accepted] *= -1


def monte_carlo_sweep(spins, temperature):
    """
    Perform one full Monte Carlo sweep.

    One sweep updates both checkerboard sublattices.
    """

    update_sublattice(
        spins,
        temperature,
        black_mask
    )

    update_sublattice(
        spins,
        temperature,
        white_mask
    )


# ============================================================
# PHYSICAL OBSERVABLES
# ============================================================

def magnetization_per_spin(spins):
    """
    Return signed magnetization per spin.
    """

    return np.mean(spins)


def energy_per_spin(spins):
    """
    Return energy per spin.

    Each bond is counted once by including only interactions
    with the right and lower neighbors.
    """

    right_neighbors = np.roll(
        spins,
        -1,
        axis=1
    )

    lower_neighbors = np.roll(
        spins,
        -1,
        axis=0
    )

    total_energy = -J * np.sum(
        spins
        * (
            right_neighbors
            + lower_neighbors
        )
    )

    return total_energy / spins.size


# ============================================================
# EXACT INFINITE-LATTICE MAGNETIZATION
# ============================================================

def exact_magnetization(temperature):
    """
    Exact spontaneous magnetization of the infinite 2D
    square-lattice Ising model at zero field.

    It is nonzero only below the exact critical temperature.
    """

    if temperature >= TC_EXACT:
        return 0.0

    sinh_value = math.sinh(
        2.0 * J / (KB * temperature)
    )

    expression = 1.0 - sinh_value ** (-4.0)

    # Protect against tiny negative roundoff errors.
    expression = max(expression, 0.0)

    return expression ** 0.125


# ============================================================
# SIMULATION AT ONE TEMPERATURE
# ============================================================

def simulate_temperature(temperature):
    """
    Run an independent simulation at one fixed temperature.

    Steps:
      1. Create a random lattice.
      2. Equilibrate the system.
      3. Collect measurements.
      4. Calculate ensemble averages and fluctuations.
    """

    spins = initialize_lattice()

    # --------------------------------------------------------
    # THERMALIZATION
    # --------------------------------------------------------

    for _ in range(THERMALIZATION_SWEEPS):
        monte_carlo_sweep(
            spins,
            temperature
        )

    # --------------------------------------------------------
    # MEASUREMENT ARRAYS
    # --------------------------------------------------------

    magnetization_samples = []
    energy_samples = []

    # --------------------------------------------------------
    # MEASUREMENT STAGE
    # --------------------------------------------------------

    for sweep in range(MEASUREMENT_SWEEPS):

        monte_carlo_sweep(
            spins,
            temperature
        )

        if (sweep + 1) % SAMPLE_INTERVAL == 0:

            magnetization_samples.append(
                magnetization_per_spin(spins)
            )

            energy_samples.append(
                energy_per_spin(spins)
            )

    magnetization_samples = np.asarray(
        magnetization_samples,
        dtype=float
    )

    energy_samples = np.asarray(
        energy_samples,
        dtype=float
    )

    # --------------------------------------------------------
    # ENSEMBLE AVERAGES
    # --------------------------------------------------------

    mean_signed_magnetization = np.mean(
        magnetization_samples
    )

    mean_absolute_magnetization = np.mean(
        np.abs(magnetization_samples)
    )

    mean_magnetization_squared = np.mean(
        magnetization_samples ** 2
    )

    mean_energy = np.mean(
        energy_samples
    )

    mean_energy_squared = np.mean(
        energy_samples ** 2
    )

    number_of_spins = spins.size

    # --------------------------------------------------------
    # MAGNETIC SUSCEPTIBILITY PER SPIN
    #
    # The absolute-magnetization estimator is useful for a
    # finite zero-field lattice because the two ordered states
    # have magnetizations +M and -M.
    # --------------------------------------------------------

    susceptibility = (
        number_of_spins
        / (KB * temperature)
        * (
            mean_magnetization_squared
            - mean_absolute_magnetization ** 2
        )
    )

    # --------------------------------------------------------
    # HEAT CAPACITY PER SPIN
    #
    # energy_samples already contains energy per spin.
    # Therefore, multiply its variance by N/T^2.
    # --------------------------------------------------------

    heat_capacity = (
        number_of_spins
        / (KB * temperature ** 2)
        * (
            mean_energy_squared
            - mean_energy ** 2
        )
    )

    return {
        "temperature": temperature,
        "mean_signed_magnetization": mean_signed_magnetization,
        "mean_absolute_magnetization": mean_absolute_magnetization,
        "mean_energy": mean_energy,
        "susceptibility": susceptibility,
        "heat_capacity": heat_capacity,
        "final_spins": spins.copy()
    }


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

def save_results_to_csv(results):
    """
    Save all numerical results to a CSV file.
    """

    with open(
        CSV_FILENAME,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "Temperature",
                "Mean_Signed_Magnetization_Per_Spin",
                "Mean_Absolute_Magnetization_Per_Spin",
                "Mean_Energy_Per_Spin",
                "Magnetic_Susceptibility_Per_Spin",
                "Heat_Capacity_Per_Spin"
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result["temperature"],
                    result["mean_signed_magnetization"],
                    result["mean_absolute_magnetization"],
                    result["mean_energy"],
                    result["susceptibility"],
                    result["heat_capacity"]
                ]
            )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 68)
    print("SCIENTIFIC 2D ISING MODEL PHASE-TRANSITION SIMULATION")
    print("=" * 68)

    print(f"Lattice size                 : {L} x {L}")
    print(f"Number of spins              : {L * L}")
    print(f"Thermalization sweeps        : {THERMALIZATION_SWEEPS}")
    print(f"Measurement sweeps           : {MEASUREMENT_SWEEPS}")
    print(f"Measurement interval         : {SAMPLE_INTERVAL}")
    print(f"Number of temperatures       : {len(temperatures)}")
    print(f"Exact critical temperature   : {TC_EXACT:.6f}")
    print(f"Random seed                  : {RANDOM_SEED}")

    print()
    print("The simulation is starting.")
    print("Progress will be displayed for every temperature.")
    print()

    start_time = time.perf_counter()

    results = []

    total_temperatures = len(temperatures)

    for temperature_index, temperature in enumerate(
        temperatures,
        start=1
    ):

        temperature_start = time.perf_counter()

        print(
            f"[{temperature_index:02d}/{total_temperatures:02d}] "
            f"T = {temperature:.6f} ... ",
            end="",
            flush=True
        )

        result = simulate_temperature(
            temperature
        )

        results.append(result)

        temperature_time = (
            time.perf_counter()
            - temperature_start
        )

        print(
            f"|M| = "
            f"{result['mean_absolute_magnetization']:.5f}   "
            f"E/N = "
            f"{result['mean_energy']:.5f}   "
            f"chi = "
            f"{result['susceptibility']:.5f}   "
            f"Cv = "
            f"{result['heat_capacity']:.5f}   "
            f"({temperature_time:.1f} s)"
        )

    total_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # CONVERT RESULTS TO NUMPY ARRAYS
    # --------------------------------------------------------

    temperature_values = np.array(
        [
            result["temperature"]
            for result in results
        ]
    )

    absolute_magnetization_values = np.array(
        [
            result["mean_absolute_magnetization"]
            for result in results
        ]
    )

    energy_values = np.array(
        [
            result["mean_energy"]
            for result in results
        ]
    )

    susceptibility_values = np.array(
        [
            result["susceptibility"]
            for result in results
        ]
    )

    heat_capacity_values = np.array(
        [
            result["heat_capacity"]
            for result in results
        ]
    )

    exact_magnetization_values = np.array(
        [
            exact_magnetization(temperature)
            for temperature in temperature_values
        ]
    )

    # --------------------------------------------------------
    # ESTIMATED PSEUDO-CRITICAL TEMPERATURES
    # --------------------------------------------------------

    susceptibility_peak_index = np.argmax(
        susceptibility_values
    )

    heat_capacity_peak_index = np.argmax(
        heat_capacity_values
    )

    susceptibility_peak_temperature = (
        temperature_values[
            susceptibility_peak_index
        ]
    )

    heat_capacity_peak_temperature = (
        temperature_values[
            heat_capacity_peak_index
        ]
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    save_results_to_csv(results)

    # --------------------------------------------------------
    # CREATE SCIENTIFIC PLOTS
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(14, 15)
    )

    ax_magnetization = axes[0, 0]
    ax_energy = axes[0, 1]
    ax_susceptibility = axes[1, 0]
    ax_heat_capacity = axes[1, 1]
    ax_low_temperature = axes[2, 0]
    ax_high_temperature = axes[2, 1]

    # --------------------------------------------------------
    # MAGNETIZATION PHASE-TRANSITION CURVE
    # --------------------------------------------------------

    ax_magnetization.plot(
        temperature_values,
        absolute_magnetization_values,
        "o-",
        color="blue",
        markersize=4,
        linewidth=1.5,
        label="Monte Carlo: <|m|>"
    )

    ax_magnetization.plot(
        temperature_values,
        exact_magnetization_values,
        "--",
        color="black",
        linewidth=2,
        label="Exact infinite-lattice result"
    )

    ax_magnetization.axvline(
        TC_EXACT,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Exact Tc = {TC_EXACT:.4f}"
    )

    ax_magnetization.set_title(
        "Magnetization Phase-Transition Curve"
    )

    ax_magnetization.set_xlabel(
        "Temperature T"
    )

    ax_magnetization.set_ylabel(
        "Average absolute magnetization <|m|>"
    )

    ax_magnetization.set_ylim(
        -0.03,
        1.05
    )

    ax_magnetization.grid(True)
    ax_magnetization.legend()

    # --------------------------------------------------------
    # ENERGY
    # --------------------------------------------------------

    ax_energy.plot(
        temperature_values,
        energy_values,
        "o-",
        color="green",
        markersize=4,
        linewidth=1.5
    )

    ax_energy.axvline(
        TC_EXACT,
        color="red",
        linestyle=":",
        linewidth=2
    )

    ax_energy.set_title(
        "Mean Energy per Spin"
    )

    ax_energy.set_xlabel(
        "Temperature T"
    )

    ax_energy.set_ylabel(
        "Average energy <e>"
    )

    ax_energy.grid(True)

    # --------------------------------------------------------
    # MAGNETIC SUSCEPTIBILITY
    # --------------------------------------------------------

    ax_susceptibility.plot(
        temperature_values,
        susceptibility_values,
        "o-",
        color="purple",
        markersize=4,
        linewidth=1.5
    )

    ax_susceptibility.axvline(
        TC_EXACT,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Exact Tc = {TC_EXACT:.4f}"
    )

    ax_susceptibility.axvline(
        susceptibility_peak_temperature,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=(
            "Susceptibility peak: "
            f"T = {susceptibility_peak_temperature:.4f}"
        )
    )

    ax_susceptibility.set_title(
        "Magnetic Susceptibility"
    )

    ax_susceptibility.set_xlabel(
        "Temperature T"
    )

    ax_susceptibility.set_ylabel(
        "Susceptibility chi"
    )

    ax_susceptibility.grid(True)
    ax_susceptibility.legend()

    # --------------------------------------------------------
    # HEAT CAPACITY
    # --------------------------------------------------------

    ax_heat_capacity.plot(
        temperature_values,
        heat_capacity_values,
        "o-",
        color="darkorange",
        markersize=4,
        linewidth=1.5
    )

    ax_heat_capacity.axvline(
        TC_EXACT,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Exact Tc = {TC_EXACT:.4f}"
    )

    ax_heat_capacity.axvline(
        heat_capacity_peak_temperature,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=(
            "Heat-capacity peak: "
            f"T = {heat_capacity_peak_temperature:.4f}"
        )
    )

    ax_heat_capacity.set_title(
        "Heat Capacity per Spin"
    )

    ax_heat_capacity.set_xlabel(
        "Temperature T"
    )

    ax_heat_capacity.set_ylabel(
        "Heat capacity Cv"
    )

    ax_heat_capacity.grid(True)
    ax_heat_capacity.legend()

    # --------------------------------------------------------
    # FINAL LOW-TEMPERATURE CONFIGURATION
    # --------------------------------------------------------

    low_temperature_result = results[0]

    low_image = ax_low_temperature.imshow(
        low_temperature_result["final_spins"],
        cmap="gray",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )

    ax_low_temperature.set_title(
        "Ordered Phase\n"
        f"T = {low_temperature_result['temperature']:.3f}"
    )

    ax_low_temperature.set_xlabel(
        "Lattice column"
    )

    ax_low_temperature.set_ylabel(
        "Lattice row"
    )

    figure.colorbar(
        low_image,
        ax=ax_low_temperature,
        fraction=0.046,
        pad=0.04
    )

    # --------------------------------------------------------
    # FINAL HIGH-TEMPERATURE CONFIGURATION
    # --------------------------------------------------------

    high_temperature_result = results[-1]

    high_image = ax_high_temperature.imshow(
        high_temperature_result["final_spins"],
        cmap="gray",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )

    ax_high_temperature.set_title(
        "Disordered Phase\n"
        f"T = {high_temperature_result['temperature']:.3f}"
    )

    ax_high_temperature.set_xlabel(
        "Lattice column"
    )

    ax_high_temperature.set_ylabel(
        "Lattice row"
    )

    figure.colorbar(
        high_image,
        ax=ax_high_temperature,
        fraction=0.046,
        pad=0.04
    )

    figure.suptitle(
        (
            f"2D Ising Model, L = {L}, "
            f"J = {J}, h = 0"
        ),
        fontsize=16
    )

    plt.tight_layout(
        rect=[0.0, 0.0, 1.0, 0.97]
    )

    figure.savefig(
        FIGURE_FILENAME,
        dpi=300,
        bbox_inches="tight"
    )

    # --------------------------------------------------------
    # FINAL TEXT OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 68)
    print("SIMULATION COMPLETE")
    print("=" * 68)

    print(
        f"Exact infinite-lattice Tc          : "
        f"{TC_EXACT:.6f}"
    )

    print(
        f"Susceptibility-peak temperature    : "
        f"{susceptibility_peak_temperature:.6f}"
    )

    print(
        f"Heat-capacity-peak temperature     : "
        f"{heat_capacity_peak_temperature:.6f}"
    )

    print(
        f"Total execution time               : "
        f"{total_time:.1f} seconds"
    )

    print(
        f"Numerical results saved to          : "
        f"{CSV_FILENAME}"
    )

    print(
        f"Scientific figure saved to          : "
        f"{FIGURE_FILENAME}"
    )

    print()
    print(
        "Note: A finite lattice does not have a perfectly sharp "
        "transition."
    )

    print(
        "Its susceptibility and heat-capacity peaks may be shifted "
        "from the exact infinite-lattice Tc."
    )

    print()
    print(
        "Close the graph window to return to the program."
    )

    plt.show()

    input(
        "\nPress Enter to close the program..."
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()