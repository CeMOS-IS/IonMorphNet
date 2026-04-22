# label_gui.py
"""
A GUI for labeling ion images from an imzML file.
Users can label images with predefined label categories.
Labels are saved to a CSV file.
"""
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import m2aia
import sys
from pathlib import Path
from msianalyzer.config import LABELS

from msianalyzer.utils import msi_utils

class IonImageLabeler:
    def __init__(self, imzml_path, output_csv, dataset_id,
                 ppm=3, labels=LABELS,
                 metadata_dict=None, only_unlabeled_mode=False):
        self.imzml_path = imzml_path
        self.output_csv = output_csv
        self.dataset_id = dataset_id
        self.ppm = ppm
        self.labels = labels
        try:
            print(f"Attempting to initialize ImzMLReader for {imzml_path}")
            self.reader = m2aia.ImzMLReader(imzml_path)
            print("ImzMLReader initialized successfully")
        except Exception as e:
            print(f"Error during ImzMLReader initialization: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
        self.index = 0
        self.metadata_dict = metadata_dict or {}
        self.counter_label_assignments = {label:0 for label in self.labels}
        self.dataset_metadata = metadata_dict.pop('_metadata', None)
        self.only_unlabeled_mode = only_unlabeled_mode

        # Create a list of (mz, msm) tuples from metadata_dict
        mz_msm_list = [(data['mz'], data.get('msm', 0)) 
                      for data in metadata_dict.values() 
                      if isinstance(data, dict) and 'mz' in data]
        
        # Sort by MSM score (descending) and m/z (ascending)
        mz_msm_list.sort(key=lambda x: (-x[1], x[0]))
        
        # Extract sorted m/z values and filter out already labeled ones
        self.mz_list = []
        labeled_mz = set()
        
        # First, check existing labels in the CSV file if it exists
        output_path = Path(self.output_csv)
        if output_path.exists():
            try:
                with output_path.open('r') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) > 1:  # Ensure row has at least mz value
                            try:
                                labeled_mz.add(round(float(row[1]), 9))  # Round to match mz_key precision
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                print(f"Warning: Could not read existing labels: {e}")
        
        # Only include m/z values that haven't been labeled yet
        if(self.only_unlabeled_mode):
            for mz, _ in mz_msm_list:
                mz_rounded = round(mz, 9)
                if mz_rounded not in labeled_mz:
                    self.mz_list.append(mz)
                    labeled_mz.add(mz_rounded)  # Add to set to prevent duplicates
        else:
            self.mz_list = [mz for mz, _ in mz_msm_list]
        print(f"Found {len(self.mz_list)} m/z values out of {len(mz_msm_list)} total")
        
        # Create a mapping from m/z to metadata key (rounded m/z)
        self.mz_to_key = {data['mz']: mz_key for mz_key, data in self.metadata_dict.items() 
                         if isinstance(data, dict) and 'mz' in data}
        
        self.cbar = None  # Initialize colorbar attribute
        self.im = None    # Initialize image attribute
        self._setup_csv()
        self._init_plot()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _setup_csv(self):
        self.labels_dict = {}  # Store labels in memory for quick access
        self.csv_entries = []  # Keep track of all entries in order
        
        # Ensure the output directory exists
        output_path = Path(self.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing labels if file exists
        if output_path.exists():
            try:
                with output_path.open('r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    headers = next(reader, None)
                    if headers != ["filename", "mz", "min_fdr", "databases", "label"]:
                        print(f"Warning: Invalid CSV header in {output_path}, starting fresh")
                        self.csv_entries = []
                    else:
                        for row in reader:
                            if len(row) >= 5:  # Ensure we have all columns
                                try:
                                    mz = float(row[1])
                                    self.labels_dict[round(mz, 9)] = row[4]  # Store label by m/z
                                    self.csv_entries.append(row)
                                except (ValueError, IndexError) as e:
                                    print(f"Warning: Skipping invalid row in {output_path}: {row}")
            except Exception as e:
                print(f"Error reading existing CSV {output_path}: {e}")
                self.csv_entries = []
        
        # Open file for writing (this will create the file if it doesn't exist)
        try:
            self.csvfile = output_path.open('w', newline='')
            self.writer = csv.writer(self.csvfile)
            self.writer.writerow(["filename", "mz", "min_fdr", "databases", "label"])
            
            # Write all existing entries
            for row in self.csv_entries:
                self.writer.writerow(row)
            self.csvfile.flush()
        except Exception as e:
            print(f"Error writing to CSV {output_path}: {e}")
            raise

    def _init_plot(self):
        # Create a figure with a better layout
        self.fig = plt.figure(figsize=(20, 12))
        self._typing_in_mz_box = False
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_click)

        # Create main layout with 1 row, 2 columns (image + metadata)
        gs = self.fig.add_gridspec(2, 2, 
                                 height_ratios=[0.85, 0.15],  # 85% for content, 15% for buttons
                                 width_ratios=[0.7, 0.3],   # 70% for image, 30% for metadata
                                 hspace=0.05, wspace=0.1)
        
        # Main image axis (left)
        self.ax = self.fig.add_subplot(gs[0, 0])
        
        # Metadata panel (right)
        self.metadata_ax = self.fig.add_subplot(gs[0, 1])
        self.metadata_ax.axis('off')
        self.metadata_text = self.metadata_ax.text(0.05, 0.95, '', 
                                                 transform=self.metadata_ax.transAxes, 
                                                 verticalalignment='top', 
                                                 fontsize=11, 
                                                 wrap=True)
        
        # Set up keyboard shortcuts
        self.keymap = {str(i+1): label for i, label in enumerate(self.labels)}
        self.keymap.update({
            'b': '_back_',
            'left': '_back_',
            'right': '_forward_',
            'n': '_forward_',
            'enter': '_forward_',
            'j': '_jump_to_mz_',
        })
        
        # Initialize buttons lists
        self.buttons = []       # For label buttons
        self.nav_buttons = []   # For navigation buttons
        self.util_buttons = []  # For utility buttons
        
        # Button dimensions
        btn_width = 0.12
        btn_margin = 0.02
        
        # Add open webpage button (utility)
        webpage_ax = self.fig.add_axes([0.02, 0.9, 0.08, 0.05])
        webpage_btn = Button(webpage_ax, "Open Metaspace", color='lightgray', hovercolor='lightblue')
        webpage_btn.on_clicked(lambda event: self._open_webpage())
        self.util_buttons.append(webpage_btn)

        # Add text box for jumping to specific m/z value
        mz_input_ax = self.fig.add_axes([0.6, 0.9, 0.15, 0.05])
        self.mz_text_box = TextBox(mz_input_ax, 'm/z:', initial='')
        self.mz_text_box.on_submit(self._jump_to_mz)

        # Add text box for jumping to index (new)
        idx_input_ax = self.fig.add_axes([0.78, 0.90, 0.08, 0.05])
        self.idx_text_box = TextBox(idx_input_ax, 'idx:', initial='')
        self.idx_text_box.on_submit(self._on_idx_submit)

        # Add back button (navigation)
        back_ax = self.fig.add_axes([0.02, 0.02, 0.08, 0.05])
        back_btn = Button(back_ax, "← Back", color='lightgray', hovercolor='lightblue')
        back_btn.on_clicked(lambda event: self._go_to_previous())
        self.nav_buttons.append(back_btn)

        # Add forward button (navigation)
        forward_ax = self.fig.add_axes([0.9, 0.02, 0.08, 0.05])
        forward_btn = Button(forward_ax, "Forward →", color='lightgray', hovercolor='lightblue')
        forward_btn.on_clicked(lambda event: self._go_to_next())
        self.nav_buttons.append(forward_btn)
        
        # Add label buttons
        total_width = len(self.labels) * btn_width + (len(self.labels) - 1) * btn_margin
        start_x = 0.5 - total_width / 2  # Center the buttons
        
        for i, label in enumerate(self.labels):
            key = str(i + 1)
            btn_ax = self.fig.add_axes([start_x + i * (btn_width + btn_margin), 0.02, btn_width, 0.05])
            
            # Check if current m/z has this label
            current_mz = self.mz_list[self.index] if self.index < len(self.mz_list) else None
            current_mz_rounded = round(current_mz, 9) if current_mz else None
            has_this_label = (current_mz_rounded in self.labels_dict and 
                           self.labels_dict[current_mz_rounded] == label)
            
            btn = Button(btn_ax, f"{key} {label.title()}", 
                        color='lightgreen' if has_this_label else 'lightgray',
                        hovercolor='lightblue')
            btn.on_clicked(self._make_label_callback(label))
            self.buttons.append(btn)
        
        # Set up keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._show_current()

        plt.show()

    def _make_label_callback(self, label):
        return lambda event: self._label(label)

    def _get_metadata_text(self, mz):
        """Generate formatted metadata text for display"""
        text_parts = []
        
        # Add dataset metadata if available
        if hasattr(self, 'dataset_metadata') and self.dataset_metadata:
            # Use the format_metadata_for_display function to format the metadata
            formatted_metadata = msi_utils.format_metadata_for_display(self.dataset_metadata)
            text_parts.extend(formatted_metadata)
            text_parts.append("")  # Add an empty line after metadata

        mz_rounded = round(mz, 9)
        md = self.metadata_dict.get(mz_rounded, {})
    
        # Get current label
        current_label = self.labels_dict.get(mz_rounded, "Not labeled yet")
        
        # Add m/z specific information
        text_parts.extend([
            f"m/z: {mz:.9f}",
            f"PPM: {self.ppm}",
            "",
            "== Current Label ==",
            current_label if current_label != "Not labeled yet" else "Not labeled yet",
            ""
        ])

        
        # Add database matches
        if 'formulas' in md and md['formulas']:
            text_parts.append("== Database Matches ==")
            for i, (formula, adduct) in enumerate(zip(md['formulas'], md['adducts'])):
                fdr = md['fdr_values'][i] if i < len(md.get('fdr_values', [])) else 1.0
                text_parts.append(f"• {formula} {adduct} (FDR: {fdr:.3f})")

        return '\n'.join(text_parts)

    def _update_button_states(self):
        """Update the button colors based on the current m/z"""
        current_mz = self.mz_list[self.index] if self.index < len(self.mz_list) else None
        current_mz_rounded = round(current_mz, 9) if current_mz else None
        
        # Reset all buttons to default color first
        for btn in self.buttons:
            btn.color = 'lightgray'
            btn.hovercolor = 'lightblue'
            btn.ax.set_facecolor('lightgray')
        
        # Set the active button to green if there's a label for current m/z
        if current_mz_rounded and current_mz_rounded in self.labels_dict:
            current_label = self.labels_dict[current_mz_rounded]
            if current_label in self.labels:
                btn_index = self.labels.index(current_label)
                if btn_index < len(self.buttons):
                    self.buttons[btn_index].color = 'lightgreen'
                    self.buttons[btn_index].ax.set_facecolor('lightgreen')
        
        self.fig.canvas.draw_idle()
    def _on_mouse_click(self, event):
        in_box = hasattr(self, 'mz_text_box') and (event.inaxes is self.mz_text_box.ax)
        self._typing_in_mz_box = bool(in_box)

        # also use matplotlib's widget lock if available
        wl = getattr(self.fig.canvas, 'widgetlock', None)
        if wl:
            try:
                if in_box:
                    # lock to textbox (API differs across mpl versions → duck-type)
                    (wl.lock if hasattr(wl, 'lock') else wl)(self.mz_text_box)
                else:
                    # release lock
                    (wl.release if hasattr(wl, 'release') else wl)(None)
            except Exception:
                pass

    def _show_current(self):
        if self.index >= len(self.mz_list):
            self.ax.clear()
            self.ax.set_title("Done! Close window.")
            self.fig.canvas.draw()
            return
            
        mz = self.mz_list[self.index]
        
        # Get the image data for the current m/z
        img_data = msi_utils.extract_ion_image(self.reader, mz, ppm=self.ppm, hotspot_removal=True)

        # Update the image data
        if hasattr(self, 'im') and self.im is not None:
            self.im.set_data(img_data)
            self.im.set_clim(vmin=img_data.min(), vmax=img_data.max())
        else:
            self.im = self.ax.imshow(img_data, cmap="viridis")
            
            # Add colorbar only once
            self.cbar = self.fig.colorbar(self.im, ax=[self.ax, self.metadata_ax])
        
        # Update title and metadata
        title = f"Ion Image {self.index + 1}/{len(self.mz_list)} - m/z {mz:.9f}"
        self.ax.set_title(title, pad=10)
        
        # Update metadata panel
        self.metadata_text.set_text(self._get_metadata_text(mz))
        
        # Update button states
        self._update_button_states()
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])
        self.fig.canvas.draw_idle()

    def _open_webpage(self):
        msi_utils.open_metaspace(self.dataset_id)

    def _go_to_previous(self):
        """Navigate to the previous image"""
        if self.index > 0:
            self.index -= 1
            self._show_current()

    def _go_to_next(self):
        """Navigate to the next image"""
        if self.index < len(self.mz_list) - 1:
            self.index += 1
            self._show_current()
    
    def _label(self, label):
        self.counter_label_assignments[label] += 1
        mz = self.mz_list[self.index]
        mz_rounded = round(mz, 9)
        md = self.metadata_dict.get(mz_rounded, {})
        min_fdr = min(md.get("fdr_values", [9.9]))
        databases = ",".join(md.get("databases", [""]))
        
        # Update in-memory storage
        self.labels_dict[mz_rounded] = label
        
        # Update or add entry in csv_entries
        entry = [os.path.basename(self.imzml_path), mz, min_fdr, databases, label]
        found = False
        for i, row in enumerate(self.csv_entries):
            if len(row) > 1 and abs(float(row[1]) - mz) < 1e-6:  # Compare m/z values
                self.csv_entries[i] = entry
                found = True
                break
        if not found:
            self.csv_entries.append(entry)
        
        # Rewrite the entire CSV file
        self.csvfile.seek(0)
        self.csvfile.truncate()
        self.writer.writerow(["filename", "mz", "min_fdr", "databases", "label"])
        self.writer.writerows(self.csv_entries)
        self.csvfile.flush()
        
        # Move to next image
        self.index += 1
        if self.index < len(self.mz_list):
            self._show_current()
        else:
            self.ax.set_title("Done! Close window.")
            self.fig.canvas.draw()

    def close(self):
        self.csvfile.close()
        return self.counter_label_assignments

    def _on_key(self, event):
        if getattr(self, '_typing_in_mz_box', False):
            return
        if event.key in self.keymap:
            action = self.keymap[event.key]
            if action == '_back_':
                self._go_to_previous()
            elif action == '_forward_':
                self._go_to_next()
            else:
                self._label(action)

    def _jump_to_mz_from_keyboard(self):
        try:
            mz_text = input("Enter m/z value to jump to: ")
            self._jump_to_mz(mz_text)
        except KeyboardInterrupt:
            print("Jump cancelled")

    def _jump_to_mz(self, text=None):
        if text is None:
            text = self.mz_text_box.text
        try:
            mz = float(text)
            # Find the closest m/z value in the list
            closest_mz = min(self.mz_list, key=lambda x: abs(x - mz))
            closest_index = self.mz_list.index(closest_mz)
            self.index = closest_index
            self._show_current()
            print(f"Jumped to m/z {closest_mz:.6f} (closest match to {mz})")
        except ValueError:
            print(f"Warning: Invalid m/z value '{text}'")

    def _on_mz_submit(self, text):
        # stop “typing” mode
        self._typing_in_mz_box = False
        wl = getattr(self.fig.canvas, 'widgetlock', None)
        if wl:
            try:
                (wl.release if hasattr(wl, 'release') else wl)(None)
            except Exception:
                pass
        # then do your actual jump
        self._jump_to_mz(text)

    def _on_idx_submit(self, text: str):
        # stop “typing” mode
        self._typing_in_mz_box = False
        wl = getattr(self.fig.canvas, 'widgetlock', None)
        if wl:
            try:
                (wl.release if hasattr(wl, 'release') else wl)(None)
            except Exception:
                pass
        self._jump_to_index(text)

    def _jump_to_index(self, text: str):
        s = text.strip()
        if not s:
            return
        try:
            i = int(s)
            # accept 1-based input; 0 and negatives clamp to 0
            target = i - 1 if i > 0 else i
            target = max(0, min(target, len(self.mz_list) - 1))
            self.index = target
            self._show_current()
        except ValueError:
            print(f"Invalid index input: {text}")


if __name__ == "__main__":
    import sys
    print("Error: This script is not meant to be run directly. Use label_workflow_gui.py instead.")
    sys.exit(1)
