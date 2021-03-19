# Measuring Energy Cost on Jetson Xavier GPU

## Introduction

This application is to profile the latency and energy cost to run a neural network on NVIDIA Jetson Xavier platform.

### Jetson AGX Power Rail

<img align=c src=https://res.cloudinary.com/dxzx2bxch/image/upload/e_bgremoval/v1616140934/rail_nlgneg.png
/>

- Xavier has two power supplies, six power rails, each has a shunt resistor.
- There are two 3-channel **INA3221** Power Monitors in the SoC, at I2C addresses `0x40`, `0x41`.
- The INA chips reports voltage, current, and power of each rail shown in the above figure. The data is accessible through Linux's sysfs nodes.
- The measurement is accurate within 5%.

More details are at the thermal design guide listed in the reference section.

### This application

This application serves as an example of profiling both latency and energy cost on the Xavier platform. 

- `config_list.pkl`: a list of neural network configurations to be profiled.
- `main.py`: run profiling. This is a multithreading application that measures latency and power at the same time. The result are written to text file automatically. 
- `read_power.py`: read power data from sysfs node.

## Dependencies
```
ofa==0.0.4 # to generate test models
torch
tqdm
```

## Run

To run the application, simply:
```
$ python main.py
```

- The actual sysfs node of INA power monitor chip's I2C bus interface might differ from device to device, so we might need to modify `filename` path in `read_power.py` to the specific sysfs path

- Use `dynamic_power(model, input_shape)` to profile power for any pytorch model.




## References
- [Jetson AGX Xavier Thermal Design Guide](https://developer.nvidia.com/embedded/dlc/jetson-agx-xavier-series-thermal-design-guide)