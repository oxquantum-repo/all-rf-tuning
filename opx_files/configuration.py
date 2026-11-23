import numpy as np
from qualang_tools.units import unit
from qm.qua._dsl import QuaVariable, QuaExpression
from qm.qua import declare, assign, play, fixed, Cast, amp, wait, ramp, ramp_to_zero
from typing import Union

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)

######################
# Network parameters #
######################
qop_ip = "10.21.64.13"  # Write the QM router IP address
cluster_name = "Cluster_1"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

# Path to save data
octave_config = None


#############################################
#              OPX PARAMETERS               #
#############################################
class OPX_virtual_gate_sequence:
    def __init__(self, configuration: dict, elements: list):
        """Framework allowing to design an arbitrary pulse sequence using virtual gates and pre-defined point from the
        charge stability map.

        :param configuration: The OPX configuration.
        :param elements: List containing the elements taking part in the virtual gate.
        """
        # List of the elements involved in the virtual gates
        self._elements = elements
        # The OPX configuration
        self._config = configuration
        # Initialize the current voltage level for sticky elements
        self.current_level = [0.0 for _ in self._elements]
        # Relevant voltage points in the charge stability diagram
        self._voltage_points = {}
        # Keep track of the averaged voltage played for defining the compensation pulse at the end of the sequence
        self.average_power = [0 for _ in self._elements]
        self._expression = None
        self._expression2 = None
        # Add to the config the step operation (length=16ns & amp=0.25V)
        for el in self._elements:
            self._config["elements"][el]["operations"]["step"] = "step_pulse"
        self._config["pulses"]["step_pulse"] = {
            "operation": "control",
            "length": 16,
            "waveforms": {"single": "step_wf"},
        }
        self._config["waveforms"]["step_wf"] = {"type": "constant", "sample": 0.25}

    def _check_name(self, name, key):
        """
        Check if a name already exists in a dictionary and append '%' until it's unique.

        :param name: The name to check.
        :param key: The dictionary to check against.
        :return: A unique name that doesn't exist in the dictionary.
        """
        if name in key:
            return self._check_name(name + "%", key)
        else:
            return name

    def _add_op_to_config(self, el: str, name: str, amplitude: float, length: int) -> str:
        """Add an operation to an element when the amplitude is fixed to release the number of real-time operations on
        the OPX.

        :param el: the element to which we want to add the operation.
        :param name: name of the operation.
        :param amplitude: Amplitude of the pulse in V.
        :param length: Duration of the pulse in ns.
        :return : The name of the created operation.
        """
        op_name = self._check_name(name, self._config["elements"][el]["operations"])
        pulse_name = self._check_name(f"{el}_{op_name}_pulse", self._config["pulses"])
        wf_name = self._check_name(f"{el}_{op_name}_wf", self._config["waveforms"])
        self._config["elements"][el]["operations"][op_name] = pulse_name
        self._config["pulses"][pulse_name] = {
            "operation": "control",
            "length": length,
            "waveforms": {"single": wf_name},
        }
        self._config["waveforms"][wf_name] = {"type": "constant", "sample": amplitude}
        return op_name


    @staticmethod
    def _check_duration(duration: int):
        """
        Check if the provided duration is valid (larger than 16 ns).

        :param duration: The duration to check in ns.
        :raises AssertionError: If the duration is less than 4 clock cycles (16 ns).
        """
        if duration is not None and not isinstance(duration, (QuaVariable, QuaExpression)):
            assert duration >= 4, "The duration must be a larger than 16 ns."

    def _update_averaged_power(self, level, duration, ramp_duration=None, current_level=None):
        """
        Update the average power calculation based on the level, duration, and optional ramp parameters.

        :param level: The voltage level in V.
        :param duration: The duration in ns.
        :param ramp_duration: Optional, the duration of the ramp in ns.
        :param current_level: Optional, the current voltage level in V.
        :return: The updated average power value.
        """
        if self.is_QUA(level):
            self._expression = declare(fixed)
            assign(self._expression, level)
            new_average = Cast.mul_int_by_fixed(duration, self._expression)
        elif self.is_QUA(duration):
            new_average = Cast.mul_int_by_fixed(duration, float(level))
        else:
            new_average = int(np.round(level * duration))

        if ramp_duration is not None:
            if not self.is_QUA(ramp_duration):
                if self.is_QUA(level):
                    self._expression2 = declare(fixed)
                    assign(self._expression2, (self._expression + current_level) >> 1)
                    new_average += Cast.mul_int_by_fixed(ramp_duration, self._expression2)
                elif self.is_QUA(current_level):
                    expression2 = declare(fixed)
                    assign(expression2, (level + current_level) >> 1)
                    new_average += Cast.mul_int_by_fixed(ramp_duration, expression2)
                elif self.is_QUA(duration):
                    new_average += Cast.mul_int_by_fixed(ramp_duration, (level + current_level) / 2)
                else:
                    new_average += int(np.round((level + current_level) * ramp_duration / 2))

            else:
                pass
        return new_average


    @staticmethod
    def is_QUA(var):
        """
        Check if a variable is a QUA variable or expression.

        :param var: The variable to check.
        :return: True if the variable is a QUA variable or expression, False otherwise.
        """
        return isinstance(var, (QuaVariable, QuaExpression))

    def add_step(
        self,
        level: list[Union[int, QuaExpression, QuaVariable]] = None,
        duration: Union[int, QuaExpression, QuaVariable] = None,
        voltage_point_name: str = None,
        ramp_duration: Union[int, QuaExpression, QuaVariable] = None,
    ) -> None:
        """Add a voltage level to the pulse sequence.
        The voltage level is either identified by its voltage_point_name if added to the voltage_point dict beforehand, or by its level and duration.
        A ramp_duration can be used to ramp to the desired level instead of stepping to it.

        :param level: Desired voltage level of the different gates composing the virtual gate in Volt.
        :param duration: How long the voltage level should be maintained in ns. Must be a multiple of 4ns and larger than 16ns.
        :param voltage_point_name: Name of the voltage level if added to the list of relevant points in the charge stability map.
        :param ramp_duration: Duration in ns of the ramp if the voltage should be ramped to the desired level instead of stepped. Must be a multiple of 4ns and larger than 16ns.
        """
        self._check_duration(duration)
        self._check_duration(ramp_duration)
        if level is not None:
            if type(level) is not list or len(level) != len(self._elements):
                raise TypeError(
                    "the provided level must be a list of same length as the number of elements involved in the virtual gate."
                )

        if voltage_point_name is not None and duration is None:
            _duration = self._voltage_points[voltage_point_name]["duration"]
        elif duration is not None:
            _duration = duration
        else:
            raise RuntimeError(
                "Either the voltage_point_name or the duration and desired voltage level must be provided."
            )

        for i, gate in enumerate(self._elements):
            if voltage_point_name is not None and level is None:
                voltage_level = self._voltage_points[voltage_point_name]["coordinates"][i]
            elif level is not None:
                voltage_point_name = "unregistered_value"
                voltage_level = level[i]
            else:
                raise RuntimeError(
                    "Either the voltage_point_name or the duration and desired voltage level must be provided."
                )
            # Play a step
            if ramp_duration is None:
                self.average_power[i] += self._update_averaged_power(voltage_level, _duration)

                # Dynamic amplitude change...
                if self.is_QUA(voltage_level) or self.is_QUA(self.current_level[i]):
                    # if dynamic duration --> play step and wait
                    if self.is_QUA(_duration):
                        play("step" * amp((voltage_level - self.current_level[i]) * 4), gate)
                        wait((_duration - 16) >> 2, gate)
                    # if constant duration --> new operation and play(*amp(..))
                    else:
                        operation = self._add_op_to_config(
                            gate,
                            "step",
                            amplitude=0.25,
                            length=_duration,
                        )
                        play(operation * amp((voltage_level - self.current_level[i]) * 4), gate)

                # Fixed amplitude but dynamic duration --> new operation and play(duration=..)
                elif isinstance(_duration, (QuaVariable, QuaExpression)):
                    operation = self._add_op_to_config(
                        gate,
                        voltage_point_name,
                        amplitude=voltage_level - self.current_level[i],
                        length=16,
                    )
                    play(operation, gate, duration=_duration >> 2)

                # Fixed amplitude and duration --> new operation and play()
                else:
                    operation = self._add_op_to_config(
                        gate,
                        voltage_point_name,
                        amplitude=voltage_level - self.current_level[i],
                        length=_duration,
                    )
                    play(operation, gate)

            # Play a ramp
            else:
                self.average_power[i] += self._update_averaged_power(
                    voltage_level, _duration, ramp_duration, self.current_level[i]
                )

                if not self.is_QUA(ramp_duration):
                    ramp_rate = 1 / ramp_duration
                    play(ramp((voltage_level - self.current_level[i]) * ramp_rate), gate, duration=ramp_duration >> 2)
                    wait(_duration >> 2, gate)

            self.current_level[i] = voltage_level


    def add_compensation_pulse(self, duration: int) -> None:
        """Add a compensation pulse of the specified duration whose amplitude is derived from the previous operations.

        :param duration: Duration of the compensation pulse in clock cycles (4ns). Must be larger than 4 clock cycles.
        """
        self._check_duration(duration)
        for i, gate in enumerate(self._elements):
            if not self.is_QUA(self.average_power[i]):
                compensation_amp = -self.average_power[i] / duration
                operation = self._add_op_to_config(
                    gate, "compensation", amplitude=compensation_amp - self.current_level[i], length=duration
                )
                play(operation, gate)
            else:
                operation = self._add_op_to_config(gate, "compensation", amplitude=0.25, length=duration)
                compensation_amp = declare(fixed)
                eval_average_power = declare(int)
                assign(eval_average_power, self.average_power[i])
                assign(compensation_amp, -Cast.mul_fixed_by_int(1 / duration, eval_average_power))
                play(operation * amp((compensation_amp - self.current_level[i]) * 4), gate)
            self.current_level[i] = compensation_amp


    def ramp_to_zero(self, duration: int = None):
        """Ramp all the gate voltages down to zero Volt and reset the averaged voltage derived for defining the compensation pulse.

        :param duration: How long will it take for the voltage to ramp down to 0V in clock cycles (4ns). If not
            provided, the default pulse duration defined in the configuration will be used.
        """
        for i, gate in enumerate(self._elements):
            ramp_to_zero(gate, duration)
            self.current_level[i] = 0
            self.average_power[i] = 0
        if self._expression is not None:
            assign(self._expression, 0)
        if self._expression2 is not None:
            assign(self._expression2, 0)


    def add_points(self, name: str, coordinates: list, duration: int) -> None:
        """Register a relevant voltage point.

        :param name: Name of the voltage point.
        :param coordinates: Voltage value of each gate involved in the virtual gate in V.
        :param duration: How long should the voltages be maintained at this level in ns. Must be larger than 16ns and a multiple of 4ns.
        """
        self._voltage_points[name] = {}
        self._voltage_points[name]["coordinates"] = coordinates
        self._voltage_points[name]["duration"] = duration


######################
#       READOUT      #
######################
# Reflectometry
resonator_IF = 249.37 * u.MHz
reflectometry_readout_length = 500 * u.ns
reflectometry_readout_amp = 20 * u.mV  # base to peak (half peak to peak)

# Time of flight
time_of_flight = 240 # + xxx

######################
#      DC GATES      #
######################

## Section defining the points from the charge stability map - can be done in the config
# Relevant points in the charge stability map as ["P1", "P2"] in V
level_init = [0.1, -0.1]
level_idle = [-0.1, 0.1]
level_readout = [0.0, 0.0]
level_compensation = [0.0, 0.0]

# Duration of each step in ns
duration_init = 200
duration_idle = 200
duration_readout = reflectometry_readout_length + 100
duration_compensation_pulse = 4 * u.us

# Step parameters
step_length = 1 * u.us
PL_step_amp = 0.1
PM_step_amp = 0.1
BL_step_amp = 0.1
BM_step_amp = 0.1

# Time to ramp down to zero for sticky elements, ONLY AFTER FULL PROGRAMME FINISHES, othewise ramp_to_zero should be used
hold_offset_duration = 10000 # in clock cycles
bias_tee_cut_off_frequency = 10 * u.kHz

######################
#    QUBIT PULSES    #
######################
# Durations in ns
pi_length = 32
pi_half_length = 16
# Amplitudes in V
pi_amps = [0.27, -0.27]
pi_half_amps = [0.27, -0.27]
gatePM_IF = 0 * u.MHz
gatePL_IF = 0 * u.MHz

burst_amp = 0.000 #0.0001 # V


#######

PL_pos_length = 16 * u.ns  # length of sticky pulse before next command can run
PL_neg_length = 16 * u.ns
PM_pos_length = 16 * u.ns
PM_neg_length = 16 * u.ns
PL_pos_amp = 0.1
PL_neg_amp = -0.2
PM_pos_amp = 0.2
PM_neg_amp = -0.1


#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0}, #'crosstalk': {2: -0.2333}},  # PL
                2: {"offset": 0.0}, #'crosstalk': {1: -0.3666}},  # PM
                3: {"offset": 0.0},  # BL
                4: {"offset": 0.0},  # BM
                5: {"offset": 0.0},  # Test1
                6: {"offset": 0.0},  # Test2
                7: {"offset": 0.0},  # Test3 reflectometry
                10: {"offset": 0.0},  # Sensor ohmic
            },
            "digital_outputs": {
                1: {},  # TTL for QDAC
                2: {},  # TTL for QDAC
            },
            "analog_inputs": {
                1: {"offset": 0.009503, "gain_db": 0},  # Test
                2: {"offset": 0.416038, "gain_db": 5},  # RF reflectometry input
            },
        },
    },

    "elements": {
        "PL": {
            "singleInput": {
                "port": ("con1", 1),
            },
            "intermediate_frequency": gatePL_IF,
            "operations": {
                "step": "PL_step_pulse",
                "burst": "burst_pulse",
                "pos_step": "PL_pos",
                "neg_step": "PL_neg",
            },
        },
        "PL_sticky": {
            "singleInput": {
                "port": ("con1", 1),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},  # in clock cycles
            "operations": {
                "step": "PL_step_pulse",
                "pos_step": "PL_pos",
                "neg_step": "PL_neg",

            },
        },
        "PM": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "intermediate_frequency": gatePM_IF,
            "operations": {
                "step": "PM_step_pulse",
                "burst": "burst_pulse",
                "pos_step": "PM_pos",
                "neg_step": "PM_neg",
            },
        },
        "PM_sticky": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "intermediate_frequency": gatePM_IF,
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "step": "PM_step_pulse",
                "burst": "burst_pulse",
                "pos_step": "PM_pos",
                "neg_step": "PM_neg",
            },
        },
        "PM_drive": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "intermediate_frequency": gatePM_IF,
            "operations": {
                "step": "PM_step_pulse",
                "burst": "burst_pulse",
                "pos_step": "PM_pos",
                "neg_step": "PM_neg",
            },
        },
        "BL": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "operations": {
                "step": "BL_step_pulse",
            },
        },
        "BM": {
            "singleInput": {
                "port": ("con1", 4),
            },
            "operations": {
                "step": "PM_step_pulse",
            },
        },
        "tank_circuit": {
            "singleInput": {
                "port": ("con1", 10),
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "readout": "reflectometry_readout_pulse",
                "burst": "reflectometry_pulse",
            },
            "outputs": {
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },

        "PL_test": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "operations": {
                "step": "PM_step_pulse",
            },
        },
        "PM_test": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "operations": {
                "step": "PM_step_pulse",
            },
        },
        "PL_sticky_test": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "step": "PL_step_pulse",
            },
        },
        "PM_sticky_test": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "step": "PM_step_pulse",
            },
        },
        "tank_circuit_test": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "readout": "reflectometry_readout_pulse",
                "burst": "reflectometry_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },

    "pulses": {
        "PL_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "PL_step_wf",
            },
        },
        "PL_pos": {
            "operation": "control",
            "length": PL_pos_length,
            "waveforms": {
                "single": "PL_pos_wf",
                },
            },
        "PL_neg": {
            "operation": "control",
            "length": PL_neg_length,
            "waveforms": {
                "single": "PL_neg_wf",
                },
            },
        "PM_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "PM_step_wf",
                },
            },
        "PM_pos": {
            "operation": "control",
            "length": PM_pos_length,
            "waveforms": {
                "single": "PM_pos_wf",
                },
            },
        "PM_neg": {
            "operation": "control",
            "length": PM_neg_length,
            "waveforms": {
                "single": "PM_neg_wf",
                },
            },
        "BL_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "BL_step_wf",
            },
        },
        "BM_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "BM_step_wf",
            },
        },
        "reflectometry_readout_pulse": {
            "operation": "measurement",
            "length": reflectometry_readout_length,
            "waveforms": {
                "single": "reflect_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
            },
            "digital_marker": "ON",
        },
        "reflectometry_pulse": {
            "operation": "control",
            "length": reflectometry_readout_length,
            "waveforms": {
                "single": "reflect_wf",
            },
        },
        "burst_pulse": {
            "operation": "control",
            "length": reflectometry_readout_length,
            "waveforms": {
                "single": "burst_wf",
            },
        },
    },

    "waveforms": {
        "PL_step_wf": {"type": "constant", "sample": PL_step_amp},
        "PL_pos_wf": {"type": "constant", "sample": PL_pos_amp},
        "PL_neg_wf": {"type": "constant", "sample": PL_neg_amp},
        "PM_step_wf": {"type": "constant", "sample": PM_step_amp},
        "PM_pos_wf": {"type": "constant", "sample": PM_pos_amp},
        "PM_neg_wf": {"type": "constant", "sample": PM_neg_amp},
        "BL_step_wf": {"type": "constant", "sample": BL_step_amp},
        "BM_step_wf": {"type": "constant", "sample": BM_step_amp},
        "reflect_wf": {"type": "constant", "sample": reflectometry_readout_amp},
        "burst_wf": {"type": "constant", "sample": burst_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },

    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },

    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, reflectometry_readout_length)],
            "sine": [(0.0, reflectometry_readout_length)],
        },
        "sine_weights": {
            "cosine": [(0.0, reflectometry_readout_length)],
            "sine": [(1.0, reflectometry_readout_length)],
        },
    },
}
