"""
Label Workflow GUI for MSI Data Explorer

This module provides a graphical interface for running the labeling workflow,
allowing users to input dataset ID, FDR threshold, and file paths for imzML
processing and output CSV generation.

Version: 2.2
"""
import os
from pathlib import Path
from typing import Optional, List
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback
import pandas as pd

from msianalyzer.utils import msi_utils
from msianalyzer.tools.labeling.label_gui import IonImageLabeler

from msianalyzer.config import PROCESSED_DIR, LABELING_CSV_DIR, METASPACE_CACHE_DIR

# Constants
FDR_VALUES: List[float] = [0.05, 0.1, 0.2, 0.5]
DEFAULT_OUTPUT_CSV: Optional[str] = None
DEFAULT_INITIAL_DIR_DATA: str = str(PROCESSED_DIR)
DEFAULT_INITIAL_DIR_CACHE: str = str(METASPACE_CACHE_DIR)
DEFAULT_OUTPUT_DIR_LABEL: str = str(LABELING_CSV_DIR)

class LabelWorkflowGUI:
    """Main application class for the Label Workflow GUI."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI components."""
        self.root = root
        self.root.title("Label Workflow")
        self._setup_ui()
        self._setup_bindings()
    
    def _setup_ui(self) -> None:
        """Set up the user interface components with improved layout and styling."""
        # Configure grid weights
        for i in range(3):
            self.root.columnconfigure(i, weight=1 if i == 1 else 0)
        
        # Create a frame for better organization
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.grid(row=0, column=0, columnspan=3, sticky="nsew")
        
        # Configure grid weights for the main frame
        for i in range(3):
            main_frame.columnconfigure(i, weight=1 if i == 1 else 0)
        
        row = 0
        
        # Section header
        header = ttk.Label(main_frame, text="METASPACE Labeling Workflow", 
                         font=('Helvetica', 14, 'bold'))
        header.grid(row=row, column=0, columnspan=3, pady=(0, 20), sticky="w")
        row += 1
        
        # Input for imzML file
        ttk.Label(main_frame, text="imzML File:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        self.imzml_path_var = tk.StringVar(value="")
        self.imzml_entry = ttk.Entry(main_frame, textvariable=self.imzml_path_var, width=50)
        self.imzml_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(main_frame, text="Browse...", command=self._browse_imzml).grid(
            row=row, column=2, padx=5, pady=5, sticky="e")
        row += 1

        # Input for FDR threshold
        ttk.Label(main_frame, text="FDR Threshold:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        self.fdr_var = tk.DoubleVar(value=0.1)
        self.fdr_entry = ttk.Combobox(
            main_frame, 
            textvariable=self.fdr_var, 
            values=FDR_VALUES, 
            state="readonly",
            width=10
        )
        self.fdr_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # Input for dataset ID
        ttk.Label(main_frame, text="METASPACE Dataset ID:").grid(
            row=row, column=0, sticky="e", padx=5, pady=5)
        self.datasetid_entry = ttk.Entry(main_frame, width=50)
        self.datasetid_entry.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        row += 1

        # Input for output CSV file
        ttk.Label(main_frame, text="Output CSV:").grid(
            row=row, column=0, sticky="e", padx=5, pady=5)
        self.output_csv_var = tk.StringVar(value=DEFAULT_OUTPUT_CSV or "")
        self.output_csv_entry = ttk.Entry(main_frame, textvariable=self.output_csv_var, width=50)
        self.output_csv_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(main_frame, text="Browse...", command=self._browse_csv).grid(
            row=row, column=2, padx=5, pady=5, sticky="e")
        row += 1
        
        # Options frame for checkboxes
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=10)
        options_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10, padx=5)
        
        # Checkbox for using cached data
        self.use_cache = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, 
            text="Use cached data", 
            variable=self.use_cache
        ).pack(side="left", padx=10)

        # Checkbox for opening METASPACE
        self.open_metaspace = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame, 
            text="Open in METASPACE", 
            variable=self.open_metaspace
        ).pack(side="left", padx=10)
        
        # Checkbox for only unlabeled mode
        self.unlabeled_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame, 
            text="Only unlabeled", 
            variable=self.unlabeled_mode
        ).pack(side="left", padx=10)

        row += 1
        
        # Button frame for better button placement
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        # Start button with improved styling
        self.start_btn = ttk.Button(
            button_frame, 
            text="Start Labeling Workflow", 
            command=self._start_workflow,
            style="Accent.TButton",
            width=25
        )
        self.start_btn.pack(pady=10)
        
        # Add some padding to all children of main_frame
        for child in main_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)
            
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            padding=(5, 2)
        )
        status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.status_var.set("Ready")
        
        # Configure grid weights for root window
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def _setup_bindings(self) -> None:
        """Set up keyboard and UI bindings."""
        self.root.bind('<Return>', lambda e: self._start_workflow())
    
    def _browse_imzml(self) -> None:
        """Open file dialog to select imzML file and update related fields."""
        try:
            # Set initial directory based on current path or default
            initial_dir = DEFAULT_INITIAL_DIR_DATA
            current_path = self.imzml_path_var.get()
            if current_path and os.path.exists(os.path.dirname(current_path)):
                initial_dir = os.path.dirname(current_path)
            
            # Show file dialog
            path = filedialog.askopenfilename(
                initialdir=initial_dir,
                filetypes=[
                    ("imzML files", "*.imzML"),
                    ("All files", "*.*")
                ],
                title="Select imzML File"
            )
            
            if not path:  # User cancelled
                return
                
            # Validate file extension
            if not path.lower().endswith('.imzml'):
                self.status_var.set("Warning: Selected file doesn't have .imzML extension")
                
            # Update CSV path to use LABELING_CSV_DIR with the same filename but .csv extension
            input_filename = Path(path).stem  # Get filename without extension
            csv_filename = f"{input_filename}.csv"
            csv_path = Path(LABELING_CSV_DIR) / csv_filename
            self.output_csv_var.set(str(csv_path))
            self.imzml_path_var.set(path)
            
            # Update status
            self.status_var.set(f"Processing {os.path.basename(path)}...")
            self.root.update_idletasks()  # Update UI
            
            # Try to get METASPACE ID and metadata
            metaspace_id = msi_utils.get_metaspace_id_from_imzml(path)
            if metaspace_id:
                self.datasetid_entry.delete(0, tk.END)
                self.datasetid_entry.insert(0, metaspace_id)
                
                # Process dataset first to ensure complete cache
                try:
                    # Process dataset with progress updates
                    self.status_var.set(f"Fetching metadata for {metaspace_id}...")
                    self.root.update_idletasks()
                    
                    # Process dataset which will populate the cache
                    msi_utils.process_metaspace_dataset(
                        metaspace_id, 
                        use_cache=True,  # Use cache if available
                        progress_callback=lambda msg: self.status_var.set(f"{msg}...")
                    )
                    
                    # Now get data from the populated cache
                    data = msi_utils.get_metaspace_data(metaspace_id, use_cache=True)
                    
                    if data and 'metadata' in data:
                        metadata = data['metadata']
                        info = metadata.get('Sample_Information', {})
                        organism = info.get('Organism', 'Unknown organism').strip()
                        part = info.get('Organism_Part', 'Unknown part').strip()
                        
                        # Clean up display text
                        if not organism or organism.lower() == 'not available':
                            organism = 'Unknown organism'
                        if not part or part.lower() == 'not available':
                            part = 'Unknown part'
                            
                        display_text = f"Selected: {os.path.basename(path)}"
                        if organism != 'Unknown organism' or part != 'Unknown part':
                            display_text += f" -> {organism} {part}"
                            
                        self.status_var.set(display_text)
                    else:
                        self.status_var.set(f"Selected: {os.path.basename(path)} (No metadata available)")
                        
                except Exception as e:
                    error_msg = str(e).split('\n')[0]  # Get first line of error
                    self.status_var.set(f"Selected: {os.path.basename(path)} (Error: {error_msg})")
                    if "not found" in str(e).lower():
                        messagebox.showwarning(
                            "Dataset Not Found",
                            f"Dataset {metaspace_id} not found on METASPACE.\n"
                            "Please verify the dataset ID is correct or upload the data to METASPACE first."
                        )
            else:
                self.status_var.set(f"Selected: {os.path.basename(path)} (No METASPACE ID found)")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting imzML file: {str(e)}")
            self.status_var.set("Error selecting imzML file")
    
    def _browse_csv(self) -> None:
        """Open save dialog to select output CSV file."""
        try:
            # Ensure LABELING_CSV_DIR exists
            Path(LABELING_CSV_DIR).mkdir(parents=True, exist_ok=True)
            
            # Set initial directory to LABELING_CSV_DIR
            initial_dir = str(LABELING_CSV_DIR)
            
            # If we have a current path and it exists, use its directory
            current_path = self.output_csv_var.get()
            if current_path and os.path.exists(os.path.dirname(current_path)):
                initial_dir = os.path.dirname(current_path)
            
            # Show save dialog
            path = filedialog.asksaveasfilename(
                initialdir=initial_dir,
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ],
                title="Save Output CSV As"
            )
            
            if path:  # User selected a file
                # Ensure the extension is .csv
                if not path.lower().endswith('.csv'):
                    path += '.csv'
                self.output_csv_var.set(path)
                self.status_var.set(f"Output will be saved to: {path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting output file: {str(e)}")
            self.status_var.set("Error selecting output file")
    
    def _start_workflow(self) -> None:
        """Start the labeling workflow with the current settings."""
        # Disable UI during processing
        self._set_ui_state(disabled=True)
        self.status_var.set("Starting workflow...")
        self.root.update_idletasks()
        
        try:
            # Get input values
            datasetid = self.datasetid_entry.get().strip()
            
            # Validate FDR
            try:
                fdr = float(self.fdr_var.get())
                if not (0 < fdr < 1):
                    messagebox.showerror("Error", "FDR must be between 0 and 1.")
                    self.status_var.set("Error: Invalid FDR value")
                    return
            except ValueError:
                messagebox.showerror("Error", "FDR must be a number between 0 and 1.")
                self.status_var.set("Error: Invalid FDR format")
                return
            
            # Get file paths
            imzml_path = self.imzml_path_var.get()
            output_csv = self.output_csv_var.get()
            
            # Validate inputs
            if not all([datasetid, imzml_path, output_csv]):
                messagebox.showerror("Error", "All fields must be filled.")
                self.status_var.set("Error: Missing required fields")
                return
            
            # Validate imzML file exists
            if not os.path.exists(imzml_path):
                messagebox.showerror("Error", f"Input file not found: {imzml_path}")
                self.status_var.set("Error: Input file not found")
                return
            
            # Ensure LABELING_CSV_DIR exists
            try:
                Path(LABELING_CSV_DIR).mkdir(parents=True, exist_ok=True)
                
                # If output_csv is not in LABELING_CSV_DIR, adjust the path
                output_path = Path(output_csv)
                if str(LABELING_CSV_DIR) not in str(output_path.parent):
                    output_path = Path(LABELING_CSV_DIR) / output_path.name
                    self.output_csv_var.set(str(output_path))
                    output_csv = str(output_path)
                    
            except OSError as e:
                messagebox.showerror("Error", f"Cannot access output directory {LABELING_CSV_DIR}: {e}")
                self.status_var.set(f"Error accessing output directory: {e}")
                return
            
            # Check write permissions for output file
            try:
                with open(output_csv, 'a') as f:
                    pass
            except IOError as e:
                messagebox.showerror("Error", f"Cannot write to output file: {e}")
                self.status_var.set(f"Error: Cannot write to output file")
                return
            
            # Run the workflow in a separate thread to keep the UI responsive
            try:
                # Run the workflow in the main thread
                self.status_var.set("Starting workflow...")
                self.root.update_idletasks()  # Update UI
                
                self._run_workflow(
                    datasetid=datasetid,
                    fdr=fdr,
                    imzml_path=imzml_path,
                    output_csv=output_csv,
                    use_cache=self.use_cache.get() if hasattr(self, 'use_cache') else True,
                    open_metaspace=self.open_metaspace.get(),
                    unlabeled_mode=self.unlabeled_mode.get()
                )
                self.status_var.set("Workflow completed successfully")
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror(
                    "Error", 
                    f"An error occurred during processing:\n{error_msg}\n\n"
                    f"See console for details."
                )
                self.status_var.set(f"Error: {error_msg}")
                traceback.print_exc()
            finally:
                self._set_ui_state(disabled=False)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            traceback.print_exc()
            self._set_ui_state(disabled=False)
    
    def _set_ui_state(self, disabled: bool = False) -> None:
        """Enable or disable UI elements during processing."""
        state = 'disabled' if disabled else 'normal'
        
        # Update main controls
        self.imzml_entry.config(state=state)
        self.fdr_entry.config(state=state)
        self.datasetid_entry.config(state=state)
        self.output_csv_entry.config(state=state)
        
        # Update buttons
        for btn in [self.start_btn, 
                   self.root.nametowidget(self.imzml_entry.winfo_parent()).children['!button'],
                   self.root.nametowidget(self.output_csv_entry.winfo_parent()).children['!button']]:
            btn.config(state=state)
        
        # Update checkboxes
        for cb in [w for w in self.root.winfo_children() if isinstance(w, ttk.Checkbutton)]:
            cb.config(state=state)
        
        # Update cursor
        self.root.config(cursor='watch' if disabled else '')
    
    def _run_workflow(
        self,
        datasetid: str,
        fdr: float,
        imzml_path: str,
        output_csv: str,
        use_cache: bool = True,
        open_metaspace: bool = False,
        unlabeled_mode: bool = False
    ) -> None:
        """
        Run the labeling workflow.
        
        Args:
            datasetid: METASPACE dataset ID
            fdr: False Discovery Rate threshold (0-1)
            imzml_path: Path to input imzML file
            output_csv: Path to output CSV file
            use_cache: Whether to use cached data if available
            open_metaspace: Whether to open the dataset in METASPACE web interface
            unlabeled_mode: Whether to only label unlabeled m/z values
        """
        try:
            # Disable UI during processing
            self._set_ui_state(disabled=True)
            self.status_var.set("Starting workflow...")
            self.root.update_idletasks()
            
            if open_metaspace:
                self.status_var.set("Opening METASPACE in browser...")
                self.root.update_idletasks()
                msi_utils.open_metaspace(datasetid, directlink=True, fdr=fdr)
            
            # Load complete dataset from cache/API
            self.status_var.set("Loading METASPACE dataset...")
            self.root.update_idletasks()
            
            # Process dataset (this will use cache if available)
            merged_data = msi_utils.process_metaspace_dataset(
                datasetid, 
                use_cache=use_cache
            )
            
            if not merged_data:
                messagebox.showerror(
                    "Error", 
                    "Failed to process METASPACE dataset. Please check your connection and try again."
                )
                return
            
            # Filter data by FDR threshold
            self.status_var.set(f"Filtering data by FDR ≤ {fdr}...")
            self.root.update_idletasks()
            
            # Filter to only include m/z values with min_fdr <= fdr
            def safe_min_fdr(vals):
                if not isinstance(vals, (list, tuple)):
                    return 1.0
                cleaned = [float(x) for x in vals if x is not None and not pd.isna(x)]
                return min(cleaned) if cleaned else 1.0
            
            filtered_data = {
                k: v for k, v in merged_data.items()
                if isinstance(v, dict) and 'mz' in v and safe_min_fdr(v.get('fdr_values', [])) <= fdr
            }
            
            if not filtered_data:
                messagebox.showerror(
                    "No Data", 
                    f"No m/z values found in the dataset with FDR ≤ {fdr}. "
                    f"Try increasing the FDR threshold."
                )
                return
            
            # get data for metadata (might be inefficient, calling it before with processing)
            data, from_cache = msi_utils.get_metaspace_data(datasetid)
            if data is None:
                messagebox.showerror("Error", f"Could not load dataset {datasetid}")
                return

            # Get metadata from the data and add it to filtered_data
            if 'metadata' in data:
                filtered_data['_metadata'] = data['metadata']

            # Get PPM from metadata
            ppm = merged_data.get('_ppm', 3)
            self.status_var.set(f"Using ppm: {ppm}")
            
            # Remove _ppm from metadata to avoid processing it as a result
            if '_ppm' in merged_data:
                del merged_data['_ppm']

            # Initialize and run the labeler with filtered data
            self.status_var.set(f"Initializing labeler with {len(filtered_data)} m/z values...")
            self.root.update_idletasks()
        
            try:
                # Run the labeler in the main thread
                labeler = IonImageLabeler(
                    imzml_path=imzml_path,
                    output_csv=output_csv,
                    dataset_id=datasetid,
                    ppm=ppm,
                    metadata_dict=filtered_data,
                    only_unlabeled_mode=unlabeled_mode
                )

                label_assignments_count = labeler.close()
                
                # Format the label counts for display
                label_counts_text = "\n".join([f"- {label}: {count}" for label, count in label_assignments_count.items()])
                print(
                    "Labeling completed successfully!\n"
                    f"Processed {len(filtered_data)} m/z values.\n\n"
                    f"Label assignments:\n{label_counts_text}\n\n"
                    f"Results saved to: {output_csv}")

                messagebox.showinfo(
                    "Labeling completed successfully!",
                    f"Processed {len(filtered_data)} m/z values.\n\n"
                    f"Label assignments:\n{label_counts_text}\n\n"
                    f"Results saved to: {output_csv}"
                )
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror(
                    "Error", 
                    f"An error occurred during processing:\n{error_msg}\n\n"
                    f"See console for details."
                )
                raise
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            raise
            
        finally:
            # Always re-enable the UI
            self._set_ui_state(disabled=False)
            self.status_var.set("Ready")

def main():
    """Initialize and run the application."""
    root = tk.Tk()
    
    # Set a modern theme if available
    try:
        import ttkthemes
        style = ttkthemes.ThemedStyle()
        style.set_theme("arc")
    except ImportError:
        # Fall back to default theme
        style = ttk.Style()
    
    # Configure the style for the start button
    style.configure('Accent.TButton', 
                   font=('Helvetica', 10, 'bold'),
                   padding=10)
    
    # Configure grid layout
    root.columnconfigure(1, weight=1)
    root.rowconfigure(6, weight=1)
    
    # Create and run the application
    app = LabelWorkflowGUI(root)
    
    # Center the window on screen
    window_width = 800
    window_height = 400
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Set minimum window size
    root.minsize(700, 350)
    
    root.mainloop()

if __name__ == "__main__":
    main()
