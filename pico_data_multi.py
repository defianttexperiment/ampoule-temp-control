import csv
import pandas as pd
import os
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter, morlet
from matplotlib.widgets import RadioButtons

class MultiFileAnalyzer:
    def __init__(self):
        # Configuration parameters
        self.temp_input_is_raw_data = True
        self.show_temp_data = False
        self.use_time_as_axis = False
        self.show_peak_lines = True
        self.interval = 10
        self.half_life = 45
        self.start_time = 0
        self.end_time = 10000
        
        # List of dataset names to load
        self.file_names = [
            '0725_cool_transition_check','0725_cool_transition_check2','0725_cool_transition_check3'
        ]
        
        # Data storage for multiple files
        self.datasets = {}
        self.current_dataset = None
        
        # GUI elements
        self.fig = None
        self.axes = None
        self.radio_ax = None
        self.radio_buttons = None
        
    def load_all_datasets(self):
        """Load all datasets specified in file_names list"""
        for file_name in self.file_names:
            print(f"Loading dataset: {file_name}")
            try:
                self.load_single_dataset(file_name)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue
        
        if self.datasets:
            self.current_dataset = list(self.datasets.keys())[0]
            print(f"Loaded {len(self.datasets)} datasets: {list(self.datasets.keys())}")
            return True
        else:
            print("No datasets loaded successfully.")
            return False
        
    def load_single_dataset(self, file_name):
        """Load a single dataset - same logic as original code"""
        # Load all CSV files from directory
        directory_path = os.getcwd() + '/' + file_name
        try:
            all_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
            if not all_files:
                print("No CSV files found in the directory.")
        except:
            print("No directory found.")
        
        # Initialize data arrays
        fulltimedata = []
        fullchanneldata = []
        fullchannelbdata = []
        csv_name = file_name + 'data.csv'
        
        # Try to load existing combined data, otherwise combine files
        try:
            initial_data = pd.read_csv(csv_name)
            fulltimedata = initial_data['Time'].tolist()
            fulltimedata.pop(0)
            fullchanneldata = initial_data['Channel A'].tolist()
            fullchanneldata.pop(0)
            fullchannelbdata = initial_data['Channel B'].tolist()
            fullchannelbdata.pop(0)
        except:
            # Add time & data from every file
            for file in all_files:
                file_path = os.path.join(directory_path, file)
                initial_data = pd.read_csv(file_path)

                strtimedata = initial_data['Time'].tolist()
                strtimedata = strtimedata[2:]
                timedata = [float(s) for s in strtimedata]
                if len(fulltimedata) > 0:
                    for i in range(len(timedata)):
                        timedata[i] = timedata[i] + fulltimedata[-1]
                fulltimedata.extend(timedata)

                strchanneldata = initial_data['Channel A'].tolist()
                strchanneldata = strchanneldata[2:]
                channeldata = []
                for s in strchanneldata:
                    try:
                        channeldata.append(float(s))
                    except ValueError:
                        channeldata.append(channeldata[-1])
                fullchanneldata.extend(channeldata)

                try:
                    strchannelbdata = initial_data['Channel B'].tolist()
                    strchannelbdata = strchannelbdata[2:]
                    channelbdata = []
                    for s in strchannelbdata:
                        try:
                            channelbdata.append(float(s))
                        except ValueError:
                            channelbdata.append(channelbdata[-1])
                    fullchannelbdata.extend(channelbdata)
                except:
                    print("No Channel B data found.")
                    fullchannelbdata = [0] * len(fullchanneldata)

            finaldf = pd.DataFrame(list(zip(fulltimedata, fullchanneldata, fullchannelbdata)), 
                                 columns = ['Time', 'Channel A', 'Channel B'])
            finaldf.to_csv(csv_name, index=False)

        # Process the signal data
        processed_signal = self.process_signal_data(fulltimedata, fullchanneldata, fullchannelbdata)
        
        # Load and process temperature data
        temp_data = self.load_temperature_data(file_name)
        
        # Store dataset
        self.datasets[file_name] = {
            'signal_data': processed_signal,
            'temp_data': temp_data
        }
    
    def process_signal_data(self, fulltimedata, fullchanneldata, fullchannelbdata):
        """Process signal data - same as original"""
        # Condense channel & time data
        smchanneldata = []
        smchannelbdata = []
        smtimedata = []
        for i in range(int(len(fullchanneldata)/self.interval)):
            start = self.interval*i
            end = self.interval*i+(self.interval-1)
            smchanneldata.append(mean(fullchanneldata[start:end]))
            smchannelbdata.append(mean(fullchannelbdata[start:end]))
            smtimedata.append(fulltimedata[start])

        # Cut channel & time data to time constraints
        if smtimedata[0] < self.start_time:
            for i in range(len(smtimedata)):
                if smtimedata[i] < self.start_time:
                    continue
                else:
                    smtimedata = smtimedata[i:]
                    smchanneldata = smchanneldata[i:]
                    smchannelbdata = smchannelbdata[i:]
                    break
        if smtimedata[-1] > self.end_time:
            for i in range(len(smtimedata)):
                if smtimedata[len(smtimedata)-1-i] > self.end_time:
                    continue
                else:
                    smtimedata = smtimedata[:len(smtimedata)-1-i]
                    smchanneldata = smchanneldata[:len(smchanneldata)-1-i]
                    smchannelbdata = smchannelbdata[:len(smchannelbdata)-1-i]
                    break

        # Apply smoothing to channel data
        smoothed_data = savgol_filter(smchanneldata, window_length=101, polyorder=3)
        smoothed_b_data = savgol_filter(smchannelbdata, window_length=101, polyorder=3)

        # Find peaks
        segment1 = slice(0, len(smoothed_data)//3)
        segment2 = slice(len(smoothed_data)//3, 2*len(smoothed_data)//3)
        segment3 = slice(2*len(smoothed_data)//3, None)

        all_peaks = []
        all_troughs = []

        segments = [segment1, segment2, segment3]
        prominences = [4, 4, 4]

        for seg, prom in zip(segments, prominences):
            peaks, _ = find_peaks(smoothed_data[seg], 
                                 prominence=prom,
                                 distance=200)
            if seg.start:
                peaks += seg.start
            all_peaks.extend(peaks)

        all_peaks_times = []
        all_troughs_times = []
        for peak in all_peaks:
            all_peaks_times.append(smtimedata[peak])
        for trough in all_troughs:
            all_troughs_times.append(smtimedata[trough])

        return {
            'smtimedata': smtimedata,
            'smchanneldata': smchanneldata,
            'smchannelbdata': smchannelbdata,
            'smoothed_data': smoothed_data,
            'smoothed_b_data': smoothed_b_data,
            'all_peaks': all_peaks,
            'all_troughs': all_troughs,
            'all_peaks_times': all_peaks_times,
            'all_troughs_times': all_troughs_times
        }
    
    def load_temperature_data(self, file_name):
        """Load temperature data - same as original"""
        try:
            raw_temp_data = pd.read_csv(file_name + 'Tdata.csv')
            temp_data = raw_temp_data['TSic AIN0 (°C)'].tolist()
            original_temp_data = temp_data.copy()
            temp_time_data = raw_temp_data['Time (s)'].tolist()

            # Smooth out raw temperature data
            if self.temp_input_is_raw_data:
                print("Smoothing raw temperature data.")
                smoothing_range = 200
                placeholder_temp_data = []
                for i in range(len(temp_data)):
                    sum_numerator = 0
                    sum_denominator = 0
                    start = max(0, i-smoothing_range)
                    
                    try: 
                        for j in range(start, i+1):
                            time_diff = i - j
                            weight = 0.5**(time_diff/self.half_life)
                            sum_numerator += temp_data[j] * weight
                            sum_denominator += weight
                        
                        if sum_denominator > 0:
                            placeholder_temp_data.append(sum_numerator/sum_denominator)
                        else:
                            placeholder_temp_data.append(temp_data[i])
                            
                    except Exception as e: 
                        print(e)
                        placeholder_temp_data.append(temp_data[i])
                temp_data = placeholder_temp_data

            # Remove data outside the time window
            if temp_time_data[0] < self.start_time:
                for i in range(len(temp_time_data)):
                    if temp_time_data[i] < self.start_time:
                        continue
                    else:
                        temp_time_data = temp_time_data[i:]
                        temp_data = temp_data[i:]
                        original_temp_data = original_temp_data[i:]
                        break
            if temp_time_data[-1] > self.end_time:
                for i in range(len(temp_time_data)):
                    if temp_time_data[len(temp_time_data)-1-i] > self.end_time:
                        continue
                    else:
                        temp_time_data = temp_time_data[:len(temp_time_data)-1-i]
                        temp_data = temp_data[:len(temp_data)-1-i]
                        original_temp_data = original_temp_data[:len(original_temp_data)-1-i]
                        break

            smooth_temp_data = savgol_filter(temp_data, window_length=120, polyorder=3)

            return {
                'temp_data': temp_data,
                'temp_time_data': temp_time_data,
                'smooth_temp_data': smooth_temp_data,
                'original_temp_data': original_temp_data
            }
        except Exception as e:
            print(f"Error loading temperature data: {e}")
            # Return dummy data if temperature file not found
            return {
                'temp_data': [20.0] * 100,
                'temp_time_data': list(range(self.start_time, self.end_time, 10)),
                'smooth_temp_data': [20.0] * 100,
                'original_temp_data': [20.0] * 100
            }

    def sliding_window_peak_rate(self, all_peaks_times, time_array, window_size=1):
        """Calculate peak rate using a true sliding window approach"""
        peak_times = np.array(all_peaks_times)
        peak_rate = np.zeros_like(time_array)
        
        for i, t in enumerate(time_array):
            window_start = t - window_size/2
            window_end = t + window_size/2
            
            peaks_in_window = np.sum((peak_times >= window_start) & 
                                    (peak_times <= window_end))
            peak_rate[i] = peaks_in_window / window_size
        
        return peak_rate

    def create_timed_temp_data(self, smtimedata, temp_data):
        """Create temperature data according to time"""
        temp_index = 0
        timed_temp_data = []
        for t in smtimedata:
            # Find correct temperature index with bounds checking
            while (temp_index < len(temp_data['temp_time_data']) - 2 and 
                   temp_data['temp_time_data'][temp_index + 1] < t):
                temp_index += 1
            
            # Ensure we stay within bounds
            if temp_index < len(temp_data['temp_time_data']) - 1:
                t_diff_below = t - temp_data['temp_time_data'][temp_index]
                t_diff_above = temp_data['temp_time_data'][temp_index + 1] - t
                interpolated_temp = (temp_data['smooth_temp_data'][temp_index] * t_diff_above + 
                                   temp_data['smooth_temp_data'][temp_index + 1] * t_diff_below) / (t_diff_above + t_diff_below)
                timed_temp_data.append(interpolated_temp)
            else:
                timed_temp_data.append(temp_data['smooth_temp_data'][-1])
        
        return timed_temp_data

    def convert_times_to_temps(self, all_peaks_times, temp_data):
        """Convert times to temps"""
        all_peaks_temps = []
        for peak_time in all_peaks_times:
            peak_index = temp_data['temp_time_data'].index(min(temp_data['temp_time_data'], key=lambda x: abs(x-peak_time)))
            rough_peak_temp = temp_data['smooth_temp_data'][peak_index]
            all_peaks_temps.append(rough_peak_temp)
        return all_peaks_temps

    def create_interactive_plot(self):
        """Create interactive plot with radio buttons"""
        if not self.datasets:
            print("No datasets loaded. Please load datasets first.")
            return
        
        # Create figure with subplots - reduced height
        self.fig, ((ax1, ax5), (ax4, ax2)) = plt.subplots(2, 2, figsize=(16, 8))
        ax3 = ax2.twinx() 
        
        self.axes = {
            'signal_plot': ax1,
            'peak_rate': ax2,
            'temp_rate': ax3,
            'signal_temp': ax4,
            'temp_plot': ax5
        }
        
        # Create nicer radio button area
        radio_ax = plt.axes([0.87, 0.35, 0.13, 0.3])  # Larger, more centered
        radio_ax.set_facecolor('#f0f0f0')  # Light gray background
        
        dataset_names = list(self.datasets.keys())
        self.radio_buttons = RadioButtons(radio_ax, dataset_names)
        
        # Style the radio buttons (compatible with all matplotlib versions)
        try:
            # Try newer matplotlib API first
            for circle in self.radio_buttons.circles:
                circle.set_facecolor('white')
                circle.set_edgecolor('steelblue')
                circle.set_linewidth(2)
            # Set active button style
            self.radio_buttons.circles[0].set_facecolor('steelblue')
        except AttributeError:
            # Fallback for older matplotlib versions
            pass
        
        # Style labels (this works in all versions)
        for label in self.radio_buttons.labels:
            label.set_fontsize(10)
            label.set_color('darkblue')
        
        # Add title to radio button area
        radio_ax.text(0.5, 0.95, 'Select Dataset', transform=radio_ax.transAxes, 
                     ha='center', va='top', fontsize=12, fontweight='bold', color='darkblue')
        
        self.radio_buttons.on_clicked(self.on_dataset_change)
        
        # Plot initial dataset
        self.plot_current_dataset()
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.8, hspace=0.25, wspace=0.3)  # Reduced spacing
        plt.show()

    def on_dataset_change(self, label):
        """Handle radio button selection"""
        self.current_dataset = label
        
        # Update radio button styling (if supported)
        try:
            for i, circle in enumerate(self.radio_buttons.circles):
                if self.radio_buttons.labels[i].get_text() == label:
                    circle.set_facecolor('steelblue')
                else:
                    circle.set_facecolor('white')
        except AttributeError:
            # Older matplotlib versions don't support this styling
            pass
        
        self.plot_current_dataset()
        self.fig.canvas.draw()

    def plot_current_dataset(self):
        """Plot the currently selected dataset - same plots as original"""
        if not self.current_dataset or self.current_dataset not in self.datasets:
            return
        
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
        
        # Get current dataset
        dataset = self.datasets[self.current_dataset]
        signal_data = dataset['signal_data']
        temp_data = dataset['temp_data']
        
        # Extract data
        smtimedata = signal_data['smtimedata']
        smchanneldata = signal_data['smchanneldata']
        smchannelbdata = signal_data['smchannelbdata']
        smoothed_data = signal_data['smoothed_data']
        smoothed_b_data = signal_data['smoothed_b_data']
        all_peaks_times = signal_data['all_peaks_times']
        all_troughs_times = signal_data['all_troughs_times']
        
        # Create timed temperature data
        timed_temp_data = self.create_timed_temp_data(smtimedata, temp_data)
        
        # Convert peak times to temperatures
        all_peaks_temps = self.convert_times_to_temps(all_peaks_times, temp_data)
        all_troughs_temps = self.convert_times_to_temps(all_troughs_times, temp_data)
        
        # Left plot: actual & smoothed data
        ax1 = self.axes['signal_plot']
        ax1.plot(smtimedata, smchanneldata, label='Raw data', alpha=0.2)
        ax1.plot(smtimedata, smoothed_data, label='Smoothed data', linewidth=2)
        ax1.plot(smtimedata, smoothed_b_data, label='Scattering data', linewidth=2)
        if self.show_peak_lines:
            ax1.vlines(all_peaks_times, 0, 20, colors="grey", alpha=0.4, label='Peaks')
            ax1.vlines(all_troughs_times, 0, 20, colors="green", alpha=0.4, label='Troughs')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal Value')
        ax1.set_title(f'Signal Data - {self.current_dataset}')
        ax1.legend()

        # Bottom left plot: signal vs temperature
        ax4 = self.axes['signal_temp']
        ax4.plot(timed_temp_data, smchanneldata, label='Raw data', alpha=0.2)
        ax4.plot(timed_temp_data, smoothed_data, label='Smoothed data', linewidth=2)
        ax4.plot(timed_temp_data, smoothed_b_data, label='Scattering data', linewidth=2)
        if self.show_peak_lines:
            ax4.vlines(all_peaks_temps, 0, 20, colors="grey", alpha=0.4, label='Peaks')
            ax4.vlines(all_troughs_temps, 0, 20, colors="green", alpha=0.4, label='Troughs')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Signal Value')
        ax4.set_title('Signal Data with Detected Peaks and Troughs')

        # Calculate derivative for second plot
        b_rate_of_change = []
        for i in range(len(smoothed_b_data)):
            try:
                b_rate_of_change.append((smoothed_b_data[i+250]-smoothed_b_data[i])*100)
            except:
                continue

        # Right plot: peak density
        ax2 = self.axes['peak_rate']
        ax3 = self.axes['temp_rate']
        
        peak_rate = self.sliding_window_peak_rate(all_peaks_temps, timed_temp_data, window_size=0.1)
        if self.use_time_as_axis:
            ax2.plot(smtimedata, peak_rate, 'b-', linewidth=2)
            ax3.plot(smtimedata[250:], b_rate_of_change, label='Scattering data', linewidth=2, alpha=0.4)
            ax2.set_xlabel('Time (s))')
        else:
            ax2.plot(timed_temp_data, peak_rate, 'b-', linewidth=2)
            ax3.plot(timed_temp_data[250:], b_rate_of_change, label='Scattering data', linewidth=2, alpha=0.4)
            ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Peaks per degree')
        ax3.set_ylabel('Differential of scattering data')
        ax3.yaxis.set_label_position("right")
        ax2.set_title('Peak Density (0.1°C window)')

        # Temperature plot
        ax5 = self.axes['temp_plot']
        ax5.plot(temp_data['temp_time_data'], temp_data['original_temp_data'], 
                temp_data['temp_time_data'], temp_data['temp_data'], 
                temp_data['temp_time_data'], temp_data['smooth_temp_data'])
        ax5.set_ylabel('Temperature (°C)')
        ax5.set_xlabel('Time (s)')
        ax5.set_title('Temperature Data')

        # Debug prints
        print(f"\nDataset: {self.current_dataset}")
        print(f"Signal data range: {smtimedata[0]:.1f} to {smtimedata[-1]:.1f}")
        print(f"Temperature data range: {temp_data['temp_time_data'][0]:.1f} to {temp_data['temp_time_data'][-1]:.1f}")
        print(f"Target range: {self.start_time} to {self.end_time}")
        print(f"Number of peaks found: {len(all_peaks_times)}")
        print(f"Signal data length: {len(smtimedata)}")
        print(f"Temperature data length: {len(timed_temp_data)}")

# Usage - Simple as the original!
def main():
    analyzer = MultiFileAnalyzer()
    
    # Edit the file_names list in the class to add your datasets
    # analyzer.file_names = ['your_dataset1', 'your_dataset2', ...]
    
    if analyzer.load_all_datasets():
        analyzer.create_interactive_plot()
    else:
        print("Failed to load any datasets.")

if __name__ == "__main__":
    main()