import csv
import pandas as pd
import os
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter, morlet
from matplotlib.widgets import RadioButtons
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class MultiDatasetAnalyzer:
    def __init__(self):
        self.datasets = {}
        self.current_dataset = None
        self.fig = None
        self.axes = None
        self.radio_buttons = None
        
        # Configuration parameters
        self.config = {
            'temp_input_is_raw_data': True,
            'show_temp_data': False,
            'use_time_as_axis': False,
            'show_peak_lines': True,
            'interval': 10,
            'half_life': 25,
            'peak_prominence': 4,
            'start_time': 1900,
            'end_time': 10000
        }
    
    def setup_gui(self):
        """Create GUI for dataset management and configuration"""
        self.root = tk.Tk()
        self.root.title("Multi-Dataset Signal Analyzer")
        self.root.geometry("400x600")
        
        # Dataset management frame
        dataset_frame = ttk.LabelFrame(self.root, text="Dataset Management", padding=10)
        dataset_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(dataset_frame, text="Add Dataset", command=self.add_dataset).pack(pady=2)
        ttk.Button(dataset_frame, text="Remove Dataset", command=self.remove_dataset).pack(pady=2)
        
        # Dataset list
        self.dataset_listbox = tk.Listbox(dataset_frame, height=4)
        self.dataset_listbox.pack(fill="x", pady=5)
        self.dataset_listbox.bind('<<ListboxSelect>>', self.on_dataset_select)
        
        # Configuration frame
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # Create configuration controls
        self.create_config_controls(config_frame)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Analyze Current Dataset", command=self.analyze_current).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Compare All Datasets", command=self.compare_all).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side="right", padx=5)
        
    def create_config_controls(self, parent):
        """Create configuration parameter controls"""
        # Checkboxes
        self.temp_raw_var = tk.BooleanVar(value=self.config['temp_input_is_raw_data'])
        ttk.Checkbutton(parent, text="Temperature input is raw data", variable=self.temp_raw_var).pack(anchor="w")
        
        self.show_temp_var = tk.BooleanVar(value=self.config['show_temp_data'])
        ttk.Checkbutton(parent, text="Show temperature data", variable=self.show_temp_var).pack(anchor="w")
        
        self.time_axis_var = tk.BooleanVar(value=self.config['use_time_as_axis'])
        ttk.Checkbutton(parent, text="Use time as axis", variable=self.time_axis_var).pack(anchor="w")
        
        self.peak_lines_var = tk.BooleanVar(value=self.config['show_peak_lines'])
        ttk.Checkbutton(parent, text="Show peak lines", variable=self.peak_lines_var).pack(anchor="w")
        
        # Numeric parameters
        numeric_params = [
            ('Interval', 'interval', 1, 100),
            ('Half Life', 'half_life', 1, 100),
            ('Peak Prominence', 'peak_prominence', 1, 20),
            ('Start Time', 'start_time', 0, 50000),
            ('End Time', 'end_time', 0, 50000)
        ]
        
        self.numeric_vars = {}
        for label, key, min_val, max_val in numeric_params:
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=f"{label}:").pack(side="left")
            var = tk.IntVar(value=self.config[key])
            self.numeric_vars[key] = var
            ttk.Spinbox(frame, from_=min_val, to=max_val, textvariable=var, width=10).pack(side="right")
    
    def add_dataset(self):
        """Add a new dataset"""
        directory = filedialog.askdirectory(title="Select dataset directory")
        if directory:
            dataset_name = os.path.basename(directory)
            if dataset_name in self.datasets:
                messagebox.showerror("Error", "Dataset already exists!")
                return
            
            try:
                # Load dataset
                dataset = self.load_dataset(directory, dataset_name)
                self.datasets[dataset_name] = dataset
                self.dataset_listbox.insert(tk.END, dataset_name)
                messagebox.showinfo("Success", f"Dataset '{dataset_name}' loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def remove_dataset(self):
        """Remove selected dataset"""
        selection = self.dataset_listbox.curselection()
        if selection:
            dataset_name = self.dataset_listbox.get(selection[0])
            del self.datasets[dataset_name]
            self.dataset_listbox.delete(selection[0])
            if self.current_dataset == dataset_name:
                self.current_dataset = None
    
    def on_dataset_select(self, event):
        """Handle dataset selection"""
        selection = self.dataset_listbox.curselection()
        if selection:
            self.current_dataset = self.dataset_listbox.get(selection[0])
    
    def load_dataset(self, directory_path, file_name):
        """Load a single dataset from directory"""
        dataset = {'name': file_name, 'directory': directory_path}
        
        # Get CSV files
        all_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        if not all_files:
            raise Exception("No CSV files found in the directory.")
        
        # Load or create combined data
        csv_name = os.path.join(directory_path, file_name + 'data.csv')
        sm_csv_name = os.path.join(directory_path, file_name + 'smdata.csv')
        
        try:
            # Try to load pre-processed data
            sm_data = pd.read_csv(sm_csv_name)
            dataset['smtimedata'] = sm_data['Time'].tolist()[1:]
            dataset['smchanneldata'] = sm_data['Channel A'].tolist()[1:]
            dataset['smchannelbdata'] = sm_data['Channel B'].tolist()[1:]
        except:
            # Process raw data
            fulltimedata, fullchanneldata, fullchannelbdata = self.process_raw_files(directory_path, all_files)
            
            # Save combined data
            finaldf = pd.DataFrame(list(zip(fulltimedata, fullchanneldata, fullchannelbdata)), 
                                 columns=['Time', 'Channel A', 'Channel B'])
            finaldf.to_csv(csv_name, index=False)
            
            # Create smoothed data
            interval = self.config['interval']
            smchanneldata, smchannelbdata, smtimedata = self.condense_data(
                fullchanneldata, fullchannelbdata, fulltimedata, interval)
            
            # Save smoothed data
            smdf = pd.DataFrame(list(zip(smtimedata, smchanneldata, smchannelbdata)), 
                              columns=['Time', 'Channel A', 'Channel B'])
            smdf.to_csv(sm_csv_name, index=False)
            
            dataset['smtimedata'] = smtimedata
            dataset['smchanneldata'] = smchanneldata
            dataset['smchannelbdata'] = smchannelbdata
        
        # Load temperature data
        temp_file = os.path.join(directory_path, file_name + 'Tdata.csv')
        if os.path.exists(temp_file):
            raw_temp_data = pd.read_csv(temp_file)
            dataset['temp_data'] = raw_temp_data['TSic AIN0 (°C)'].tolist()
            dataset['temp_time_data'] = raw_temp_data['Time (s)'].tolist()
        else:
            dataset['temp_data'] = None
            dataset['temp_time_data'] = None
        
        return dataset
    
    def process_raw_files(self, directory_path, all_files):
        """Process raw CSV files and combine them"""
        fulltimedata = []
        fullchanneldata = []
        fullchannelbdata = []
        
        for file in all_files:
            file_path = os.path.join(directory_path, file)
            initial_data = pd.read_csv(file_path)
            
            # Process time data
            strtimedata = initial_data['Time'].tolist()[2:]
            timedata = [float(s) for s in strtimedata]
            if len(fulltimedata) > 0:
                for i in range(len(timedata)):
                    timedata[i] = timedata[i] + fulltimedata[-1]
            fulltimedata.extend(timedata)
            
            # Process Channel A data
            strchanneldata = initial_data['Channel A'].tolist()[2:]
            channeldata = []
            for s in strchanneldata:
                try:
                    channeldata.append(float(s))
                except ValueError:
                    channeldata.append(channeldata[-1] if channeldata else 0)
            fullchanneldata.extend(channeldata)
            
            # Process Channel B data
            try:
                strchannelbdata = initial_data['Channel B'].tolist()[2:]
                channelbdata = []
                for s in strchannelbdata:
                    try:
                        channelbdata.append(float(s))
                    except ValueError:
                        channelbdata.append(channelbdata[-1] if channelbdata else 0)
                fullchannelbdata.extend(channelbdata)
            except:
                fullchannelbdata = [0] * len(fullchanneldata)
        
        return fulltimedata, fullchanneldata, fullchannelbdata
    
    def condense_data(self, fullchanneldata, fullchannelbdata, fulltimedata, interval):
        """Condense data by averaging over intervals"""
        smchanneldata = []
        smchannelbdata = []
        smtimedata = []
        
        for i in range(int(len(fullchanneldata)/interval)):
            start = interval * i
            end = interval * i + (interval - 1)
            smchanneldata.append(mean(fullchanneldata[start:end]))
            smchannelbdata.append(mean(fullchannelbdata[start:end]))
            smtimedata.append(fulltimedata[start])
        
        return smchanneldata, smchannelbdata, smtimedata
    
    def update_config(self):
        """Update configuration from GUI"""
        self.config['temp_input_is_raw_data'] = self.temp_raw_var.get()
        self.config['show_temp_data'] = self.show_temp_var.get()
        self.config['use_time_as_axis'] = self.time_axis_var.get()
        self.config['show_peak_lines'] = self.peak_lines_var.get()
        
        for key, var in self.numeric_vars.items():
            self.config[key] = var.get()
    
    def analyze_current(self):
        """Analyze the currently selected dataset"""
        if not self.current_dataset:
            messagebox.showerror("Error", "No dataset selected!")
            return
        
        self.update_config()
        self.analyze_dataset(self.datasets[self.current_dataset])
    
    def compare_all(self):
        """Compare all loaded datasets"""
        if len(self.datasets) < 2:
            messagebox.showerror("Error", "Need at least 2 datasets for comparison!")
            return
        
        self.update_config()
        self.create_comparison_plot()
    
    def analyze_dataset(self, dataset):
        """Analyze a single dataset"""
        # Process data according to current configuration
        processed_data = self.process_dataset(dataset)
        
        # Create analysis plot
        self.create_analysis_plot(processed_data)
    
    def process_dataset(self, dataset):
        """Process dataset according to current configuration"""
        # Get data
        smtimedata = dataset['smtimedata'].copy()
        smchanneldata = dataset['smchanneldata'].copy()
        smchannelbdata = dataset['smchannelbdata'].copy()
        
        # Cut data to time constraints
        start_time = self.config['start_time']
        end_time = self.config['end_time']
        
        if smtimedata[0] < start_time:
            for i in range(len(smtimedata)):
                if smtimedata[i] >= start_time:
                    smtimedata = smtimedata[i:]
                    smchanneldata = smchanneldata[i:]
                    smchannelbdata = smchannelbdata[i:]
                    break
        
        if smtimedata[-1] > end_time:
            for i in range(len(smtimedata)):
                if smtimedata[len(smtimedata)-1-i] <= end_time:
                    smtimedata = smtimedata[:len(smtimedata)-i]
                    smchanneldata = smchanneldata[:len(smchanneldata)-i]
                    smchannelbdata = smchannelbdata[:len(smchannelbdata)-i]
                    break
        
        # Apply smoothing
        smoothed_data = savgol_filter(smchanneldata, window_length=min(101, len(smchanneldata)//2*2-1), polyorder=3)
        smoothed_b_data = savgol_filter(smchannelbdata, window_length=min(101, len(smchannelbdata)//2*2-1), polyorder=3)
        
        # Process temperature data if available
        timed_temp_data = None
        if dataset['temp_data'] is not None:
            timed_temp_data = self.process_temperature_data(dataset, smtimedata, start_time, end_time)
        
        # Find peaks
        all_peaks, _ = find_peaks(smoothed_data, prominence=self.config['peak_prominence'], distance=200)
        all_peaks_times = [smtimedata[peak] for peak in all_peaks]
        
        # Calculate peak rate if temperature data available
        peak_rate = None
        if timed_temp_data is not None:
            all_peaks_temps = []
            for peak_time in all_peaks_times:
                temp_index = min(range(len(dataset['temp_time_data'])), 
                               key=lambda i: abs(dataset['temp_time_data'][i] - peak_time))
                all_peaks_temps.append(timed_temp_data[temp_index])
            
            peak_rate = self.sliding_window_peak_rate(all_peaks_temps, timed_temp_data, window_size=0.1)
        
        return {
            'name': dataset['name'],
            'smtimedata': smtimedata,
            'smchanneldata': smchanneldata,
            'smchannelbdata': smchannelbdata,
            'smoothed_data': smoothed_data,
            'smoothed_b_data': smoothed_b_data,
            'all_peaks': all_peaks,
            'all_peaks_times': all_peaks_times,
            'timed_temp_data': timed_temp_data,
            'peak_rate': peak_rate
        }
    
    def process_temperature_data(self, dataset, smtimedata, start_time, end_time):
        """Process temperature data"""
        temp_data = dataset['temp_data'].copy()
        temp_time_data = dataset['temp_time_data'].copy()
        
        # Smooth raw temperature data if needed
        if self.config['temp_input_is_raw_data']:
            half_life = self.config['half_life']
            smoothing_range = 200
            placeholder_temp_data = []
            
            for i in range(len(temp_data)):
                sum_numerator = 0
                sum_denominator = 0
                start_idx = max(0, i - smoothing_range)
                
                for j in range(start_idx, i + 1):
                    time_diff = i - j
                    weight = 0.5**(time_diff / half_life)
                    sum_numerator += temp_data[j] * weight
                    sum_denominator += weight
                
                if sum_denominator > 0:
                    placeholder_temp_data.append(sum_numerator / sum_denominator)
                else:
                    placeholder_temp_data.append(temp_data[i])
            
            temp_data = placeholder_temp_data
        
        # Cut temperature data to time constraints
        if temp_time_data[0] < start_time:
            for i in range(len(temp_time_data)):
                if temp_time_data[i] >= start_time:
                    temp_time_data = temp_time_data[i:]
                    temp_data = temp_data[i:]
                    break
        
        if temp_time_data[-1] > end_time:
            for i in range(len(temp_time_data)):
                if temp_time_data[len(temp_time_data)-1-i] <= end_time:
                    temp_time_data = temp_time_data[:len(temp_time_data)-i]
                    temp_data = temp_data[:len(temp_data)-i]
                    break
        
        # Smooth temperature data
        smooth_temp_data = savgol_filter(temp_data, window_length=min(120, len(temp_data)//2*2-1), polyorder=3)
        
        # Interpolate temperature data to match signal time points
        temp_index = 0
        timed_temp_data = []
        
        for t in smtimedata:
            while (temp_index < len(temp_time_data) - 2 and 
                   temp_time_data[temp_index + 1] < t):
                temp_index += 1
            
            if temp_index < len(temp_time_data) - 1:
                t_diff_below = t - temp_time_data[temp_index]
                t_diff_above = temp_time_data[temp_index + 1] - t
                interpolated_temp = (smooth_temp_data[temp_index] * t_diff_above + 
                                   smooth_temp_data[temp_index + 1] * t_diff_below) / (t_diff_above + t_diff_below)
                timed_temp_data.append(interpolated_temp)
            else:
                timed_temp_data.append(smooth_temp_data[-1])
        
        return timed_temp_data
    
    def sliding_window_peak_rate(self, all_peaks_temps, timed_temp_data, window_size=1):
        """Calculate peak rate using sliding window"""
        peak_times = np.array(all_peaks_temps)
        peak_rate = np.zeros_like(timed_temp_data)
        
        for i, t in enumerate(timed_temp_data):
            window_start = t - window_size/2
            window_end = t + window_size/2
            
            peaks_in_window = np.sum((peak_times >= window_start) & 
                                   (peak_times <= window_end))
            
            peak_rate[i] = peaks_in_window / window_size
        
        return peak_rate
    
    def create_analysis_plot(self, data):
        """Create analysis plot for single dataset"""
        plt.close('all')
        fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(2, 2, figsize=(16, 10))
        ax3 = ax2.twinx()
        
        # Plot signal data
        ax1.plot(data['smtimedata'], data['smchanneldata'], label='Raw data', alpha=0.2)
        ax1.plot(data['smtimedata'], data['smoothed_data'], label='Smoothed data', linewidth=2)
        ax1.plot(data['smtimedata'], data['smoothed_b_data'], label='Scattering data', linewidth=2)
        
        if self.config['show_peak_lines']:
            ax1.vlines(data['all_peaks_times'], 0, 20, colors="grey", alpha=0.4, label='Peaks')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal Value')
        ax1.set_title(f'Signal Data - {data["name"]}')
        ax1.legend()
        
        # Plot temperature-based data if available
        if data['timed_temp_data'] is not None:
            ax5.plot(data['timed_temp_data'], data['smchanneldata'], label='Raw data', alpha=0.2)
            ax5.plot(data['timed_temp_data'], data['smoothed_data'], label='Smoothed data', linewidth=2)
            ax5.plot(data['timed_temp_data'], data['smoothed_b_data'], label='Scattering data', linewidth=2)
            ax5.set_xlabel('Temperature (°C)')
            ax5.set_ylabel('Signal Value')
            ax5.set_title('Temperature vs Signal')
            
            # Plot peak rate
            if data['peak_rate'] is not None:
                x_axis = data['smtimedata'] if self.config['use_time_as_axis'] else data['timed_temp_data']
                ax2.plot(x_axis, data['peak_rate'], 'b-', linewidth=2)
                ax2.set_xlabel('Time (s)' if self.config['use_time_as_axis'] else 'Temperature (°C)')
                ax2.set_ylabel('Peaks per degree')
                ax2.set_title('Peak Density')
        
        plt.tight_layout()
        plt.show()
    
    def create_comparison_plot(self):
        """Create comparison plot for all datasets"""
        plt.close('all')
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.datasets)))
        
        for i, (name, dataset) in enumerate(self.datasets.items()):
            processed_data = self.process_dataset(dataset)
            color = colors[i]
            
            # Plot smoothed signals
            axes[0, 0].plot(processed_data['smtimedata'], processed_data['smoothed_data'], 
                           label=f'{name} - Channel A', color=color, linewidth=2)
            axes[0, 1].plot(processed_data['smtimedata'], processed_data['smoothed_b_data'], 
                           label=f'{name} - Channel B', color=color, linewidth=2)
            
            # Plot temperature-based comparisons if available
            if processed_data['timed_temp_data'] is not None:
                axes[1, 0].plot(processed_data['timed_temp_data'], processed_data['smoothed_data'], 
                               label=f'{name}', color=color, linewidth=2)
                
                if processed_data['peak_rate'] is not None:
                    axes[1, 1].plot(processed_data['timed_temp_data'], processed_data['peak_rate'], 
                                   label=f'{name}', color=color, linewidth=2)
        
        # Set labels and titles
        axes[0, 0].set_title('Channel A Comparison')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Signal Value')
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Channel B Comparison')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Signal Value')
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Temperature vs Signal Comparison')
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Signal Value')
        axes[1, 0].legend()
        
        axes[1, 1].set_title('Peak Rate Comparison')
        axes[1, 1].set_xlabel('Temperature (°C)')
        axes[1, 1].set_ylabel('Peaks per degree')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Run the application"""
        self.setup_gui()
        self.root.mainloop()

# Usage
if __name__ == "__main__":
    analyzer = MultiDatasetAnalyzer()
    analyzer.run()