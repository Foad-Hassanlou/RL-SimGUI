# main.py
import threading
import customtkinter as ctk
from pprint import pprint
from tkinter import messagebox
import matplotlib.pyplot as plt
import pandas as pd
import os
import webbrowser
import sys
import tempfile
from PIL import Image, ImageTk
import cv2
from customtkinter import CTkImage

from src.train import Model_TrainTest
from requirements.config import get_hyperparameters
import numpy as np

from tabulate import tabulate

# Set customtkinter appearance mode
ctk.set_appearance_mode("dark")

def ensure_directories_exist(hp):
    """Make sure save/plot/video dirs exist."""
    for key in ["save_path", "plot_path", "video_path"]:
        path = hp.get(key)
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

class RLExperimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Actor-Critic vs. Value-Based: Empirical Trade-offs")
        self.root.configure(bg="#2c3e50")

        # Center the window on screen
        window_width = 1100
        window_height = 700
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        root.minsize(920, 690)

        # Valid combinations from the table
        self.valid_combinations = {
            "Discrete": {
                "DQN": ["MountainCar-v0", "FrozenLake-v1"],
                "A2C": ["MountainCar-v0", "FrozenLake-v1"]
            },
            "Continuous": {
                "NAF": ["HalfCheetah-v4", "Pendulum-v1"],
                "SAC": ["HalfCheetah-v4", "Pendulum-v1"]
            }
        }

        # Main frame
        self.main_frame = ctk.CTkFrame(root, fg_color="#34495e", corner_radius=20)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Title
        ctk.CTkLabel(
            self.main_frame,
            text="Actor-Critic vs. Value-Based",
            font=("Arial Rounded MT Bold", 21, "bold"),
            text_color="#ecf0f1",
            pady=5,
            anchor="center",
            justify="center"
        ).pack(fill="x")

        # Frame for 2x2 grid layout of selection widgets
        selection_frame = ctk.CTkFrame(self.main_frame, fg_color="#34495e")
        selection_frame.pack(pady=10, fill="x")
        selection_frame.grid_columnconfigure((0, 1), weight=1)

        # Action Space
        action_space_frame = ctk.CTkFrame(selection_frame, fg_color="#34495e")
        action_space_frame.grid(row=0, column=0, padx=5, pady=3, sticky="ee")
        ctk.CTkLabel(
            action_space_frame,
            text="Action Space:",
            text_color="#ecf0f1",
            font=("Arial Rounded MT", 12),
            anchor="center",
            justify="center"
        ).pack(fill="x")
        self.action_space_var = ctk.StringVar(value="Discrete")
        self.action_space_menu = ctk.CTkComboBox(
            action_space_frame,
            values=["Discrete", "Continuous"],
            variable=self.action_space_var,
            width=200,
            font=("Arial Rounded MT", 10),
            dropdown_font=("Arial Rounded MT", 10),
            fg_color="#ecf0f1",
            button_color="#4CAF50",
            button_hover_color="#45A049",
            text_color="#2c3e50",
            state="readonly",
            justify="center"
        )
        self.action_space_menu.pack(pady=2)
        self.action_space_menu.configure(command=self.update_algorithm_menu)

        # Algorithm
        algorithm_frame = ctk.CTkFrame(selection_frame, fg_color="#34495e")
        algorithm_frame.grid(row=0, column=1, padx=5, pady=3, sticky="ww")
        ctk.CTkLabel(
            algorithm_frame,
            text="Algorithm:",
            text_color="#ecf0f1",
            font=("Arial Rounded MT", 12),
            anchor="center",
            justify="center"
        ).pack(fill="x")
        self.algorithm_var = ctk.StringVar()
        self.algorithm_menu = ctk.CTkComboBox(
            algorithm_frame,
            values=[],
            variable=self.algorithm_var,
            width=200,
            font=("Arial Rounded MT", 10),
            dropdown_font=("Arial Rounded MT", 10),
            fg_color="#ecf0f1",
            button_color="#4CAF50",
            button_hover_color="#45A049",
            text_color="#2c3e50",
            state="readonly",
            justify="center"
        )
        self.algorithm_menu.pack(pady=2)
        self.algorithm_menu.configure(command=self.update_environment_menu)

        # Environment
        env_frame = ctk.CTkFrame(selection_frame, fg_color="#34495e")
        env_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ee")
        ctk.CTkLabel(
            env_frame,
            text="Environment:",
            text_color="#ecf0f1",
            font=("Arial Rounded MT", 12),
            anchor="center",
            justify="center"
        ).pack(fill="x")
        self.env_var = ctk.StringVar()
        self.env_menu = ctk.CTkComboBox(
            env_frame,
            values=[],
            variable=self.env_var,
            width=200,
            font=("Arial Rounded MT", 10),
            dropdown_font=("Arial Rounded MT", 10),
            fg_color="#ecf0f1",
            button_color="#4CAF50",
            button_hover_color="#45A049",
            text_color="#2c3e50",
            state="readonly",
            justify="center"
        )
        self.env_menu.pack(pady=2)
        self.env_menu.configure(command=self.update_map_size_visibility)

        # Mode
        mode_frame = ctk.CTkFrame(selection_frame, fg_color="#34495e")
        mode_frame.grid(row=1, column=1, padx=5, pady=3, sticky="ww")
        ctk.CTkLabel(
            mode_frame,
            text="Mode:",
            text_color="#ecf0f1",
            font=("Arial Rounded MT", 12),
            anchor="center",
            justify="center"
        ).pack(fill="x")
        self.mode_var = ctk.StringVar(value="train")
        self.mode_menu = ctk.CTkComboBox(
            mode_frame,
            values=["train", "test"],
            variable=self.mode_var,
            width=200,
            font=("Arial Rounded MT", 10),
            dropdown_font=("Arial Rounded MT", 10),
            fg_color="#ecf0f1",
            button_color="#4CAF50",
            button_hover_color="#45A049",
            text_color="#2c3e50",
            state="readonly",
            justify="center"
        )
        self.mode_menu.pack(pady=2)

        # Button Frame
        button_frame = ctk.CTkFrame(self.main_frame, fg_color="#34495e", corner_radius=20)
        button_frame.pack(pady=10)

        # Run Button
        self.run_button = ctk.CTkButton(
            button_frame,
            text="Run",
            command=lambda: [print("Run button clicked"), self.run_experiment()],
            fg_color="#4CAF50",
            hover_color="#45A049",
            font=("Arial Rounded MT", 10),
            corner_radius=10,
            width=120,
            text_color="white"
        )
        self.run_button.pack(side="left", padx=5)

        # Show Plot Button
        plot_button = ctk.CTkButton(
            button_frame,
            text="Show Plots",
            command=lambda: [print("Show Plots button clicked"), self.show_plots()],
            fg_color="#2196F3",
            hover_color="#1E88E5",
            font=("Arial Rounded MT", 10),
            corner_radius=10,
            width=120,
            text_color="white"
        )
        plot_button.pack(side="left", padx=5)

        # Show Video Button
        video_button = ctk.CTkButton(
            button_frame,
            text="Show Videos",
            command=lambda: [print("Show Videos button clicked"), self.show_videos()],
            fg_color="#9C27B0",
            hover_color="#8E24AA",
            font=("Arial Rounded MT", 10),
            corner_radius=10,
            width=120,
            text_color="white"
        )
        video_button.pack(side="left", padx=5)

        # Compare Button
        compare_button = ctk.CTkButton(
            button_frame,
            text="Compare",
            command=lambda: [print("Compare button clicked"), self.open_compare_dialog()],
            fg_color="#FF9800",
            hover_color="#FB8C00",
            font=("Arial Rounded MT", 10),
            corner_radius=10,
            width=120,
            text_color="white"
        )
        compare_button.pack(side="left", padx=5)

        # Show Hyperparameters Button
        hyperparameters_button = ctk.CTkButton(
            button_frame,
            text="Hyperparameters",
            command=lambda: [print("Hyperparameters button clicked"), self.show_hyperparameters()],
            fg_color="#FF5722",
            hover_color="#F4511E",
            font=("Arial Rounded MT", 10),
            corner_radius=10,
            width=120,
            text_color="white"
        )
        hyperparameters_button.pack(side="left", padx=5)

        # Structure Button to Button Frame
        structure_button = ctk.CTkButton(
            button_frame,
            text="Structure",
            command=lambda: [print("Structure button clicked"), self.show_structure()],
            fg_color="#FFCA28",
            hover_color="#FFB74D",
            font=("Arial Rounded MT", 10),
            corner_radius=10,
            width=120,
            text_color="white"
        )
        structure_button.pack(side="left", padx=5)

        # Clear Console Button
        clear_console_button = ctk.CTkButton(
            button_frame,
            text="Clear Console",
            command=lambda: [print("Clear Console button clicked"), self.clear_console()],
            fg_color="#0288D1",
            hover_color="#0277BD",
            font=("Arial Rounded MT", 10),
            corner_radius=10,
            width=120,
            text_color="white"
        )
        clear_console_button.pack(side="left", padx=5)

        self.map_size_label = ctk.CTkLabel(
            selection_frame,
            text="Map Size (FrozenLake-v1):",
            text_color="#ecf0f1",
            font=("Arial Rounded MT", 12),
            anchor="center",
            justify="center"
        )
        self.map_size_label.grid(row=2, column=0, padx=5, pady=2, sticky="ew", columnspan=2)
        self.map_size_label.grid_remove()

        self.map_size_var = ctk.StringVar(value="8")
        self.map_size_menu = ctk.CTkComboBox(
            selection_frame,
            values=["4", "8"],
            variable=self.map_size_var,
            width=200,
            font=("Arial Rounded MT", 10),
            dropdown_font=("Arial Rounded MT", 10),
            fg_color="#ecf0f1",
            button_color="#4CAF50",
            button_hover_color="#45A049",
            text_color="#2c3e50",
            state="readonly",
            justify="center"
        )
        self.map_size_menu.grid(row=3, column=0, padx=5, pady=2, columnspan=2)
        self.map_size_menu.grid_remove()

        # Console Output
        console_frame = ctk.CTkFrame(self.main_frame, fg_color="#34495e", corner_radius=20)
        console_frame.pack(pady=10, padx=20, fill="both", expand=True)
        ctk.CTkLabel(console_frame, text="Console Output:", text_color="#ecf0f1", font=("Arial Rounded MT", 12),
                     anchor="center").pack(fill="x")
        self.console_text = ctk.CTkTextbox(console_frame, height=200, fg_color="#000000", text_color="#00FF00",
                                           font=("Arial Rounded MT", 9), state="disabled", wrap="word")
        self.console_text.pack(fill="both", expand=True)
        sys.stdout = self.ConsoleRedirector(self.console_text)

        # Exit Button
        exit_button = ctk.CTkButton(
            self.main_frame,
            text="Exit",
            command=lambda: [print("Exit button clicked"), self.root.quit()],
            fg_color="#F44336",
            hover_color="#D32F2F",
            font=("Arial Rounded MT", 10),
            corner_radius=10,
            width=120,
            text_color="white"
        )
        exit_button.pack_configure(pady=20)

        # Initialize menus
        self.update_algorithm_menu()

    class ConsoleRedirector:
        """Redirect stdout to a customtkinter Textbox widget."""
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, message):
            def update_text():
                self.text_widget.configure(state="normal")
                self.text_widget.insert("end", message)
                self.text_widget.see("end")
                self.text_widget.configure(state="disabled")

            self.text_widget.after(0, update_text)

        def flush(self):
            pass

    def update_algorithm_menu(self, event=None):
        """Update algorithm menu based on selected action space."""
        action_space = self.action_space_var.get()
        algorithms = list(self.valid_combinations[action_space].keys())
        self.algorithm_menu.configure(values=algorithms)
        self.algorithm_var.set(algorithms[0] if algorithms else "")
        self.update_environment_menu()

    def update_environment_menu(self, event=None):
        """Update environment menu based on selected algorithm."""
        action_space = self.action_space_var.get()
        algorithm = self.algorithm_var.get()
        environments = self.valid_combinations[action_space].get(algorithm, [])
        self.env_menu.configure(values=environments)
        self.env_var.set(environments[0] if environments else "")
        self.update_map_size_visibility()

    def update_map_size_visibility(self, event=None):
        """Show or hide map size widgets based on selected environment."""
        if self.env_var.get() == "FrozenLake-v1":
            self.map_size_label.grid()
            self.map_size_menu.grid()
        else:
            self.map_size_label.grid_remove()
            self.map_size_menu.grid_remove()

        self.root.update_idletasks()

    def clear_console(self):
        """Clear the console text widget."""
        self.console_text.configure(state="normal")
        self.console_text.delete("0.0", "end")
        self.console_text.configure(state="disabled")

    def show_hyperparameters(self):
        """Display hyperparameters in a scrollable table and print a clean summary to console."""
        algorithm = self.algorithm_var.get().upper()
        env_name = self.env_var.get()
        mode = self.mode_var.get()

        if not algorithm or not env_name:
            messagebox.showerror("Error", "Please select valid algorithm and environment.")
            return

        hp = get_hyperparameters(algorithm, env_name, mode, int(self.map_size_var.get()))

        if env_name == "FrozenLake-v1":
            hp["map_size"] = int(self.map_size_var.get())
            hp["num_states"] = hp["map_size"] ** 2

        # ----------- Console Output (Cleaned Table) -----------
        # Clean long values for console display
        cleaned_hp = {}
        for key, val in hp.items():
            if "path" in key.lower() and isinstance(val, str):
                # Show only the basename of the path
                cleaned_hp[key] = "..."
            else:
                cleaned_hp[key] = val

        print(f"Hyperparameters for {algorithm} on {env_name} ({mode} mode):")
        print(tabulate(cleaned_hp.items(), headers=["Parameter", "Value"], tablefmt="fancy_grid"))
        print()

        # ----------- CTk Window Display -----------
        window = ctk.CTkToplevel(self.root)
        window.title(f"Hyperparameters for {algorithm} on {env_name}")
        window.geometry("890x400")
        window.focus_force()
        window.grab_set()

        # Title
        title_frame = ctk.CTkFrame(window)
        title_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(
            title_frame,
            text=f"Hyperparameters for {algorithm} ({env_name}, {mode} mode):",
            font=("Arial Rounded MT", 14),
            justify="center",
            anchor="center"
        ).pack(fill="x")

        # Scrollable canvas setup
        canvas = ctk.CTkCanvas(window)
        scrollbar = ctk.CTkScrollbar(window, orientation="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure columns
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)

        # Header row
        ctk.CTkLabel(
            scrollable_frame,
            text="Parameter",
            font=("Arial Rounded MT", 12, "bold"),
            justify="center",
            anchor="center"
        ).grid(row=0, column=0, sticky="nsew", padx=10, pady=(0, 5))
        ctk.CTkLabel(
            scrollable_frame,
            text="Value",
            font=("Arial Rounded MT", 12, "bold"),
            justify="center",
            anchor="center"
        ).grid(row=0, column=1, sticky="nsew", padx=10, pady=(0, 5))

        # Data rows
        for i, (key, value) in enumerate(hp.items(), start=1):
            ctk.CTkLabel(
                scrollable_frame,
                text=str(key),
                font=("Arial Rounded MT", 10),
                justify="center",
                anchor="center"
            ).grid(row=i, column=0, sticky="nsew", padx=10, pady=2)
            ctk.CTkLabel(
                scrollable_frame,
                text=str(value),
                font=("Arial Rounded MT", 10),
                justify="center",
                anchor="center"
            ).grid(row=i, column=1, sticky="nsew", padx=10, pady=2)

    def open_compare_dialog(self):
        """Open a dialog to compare two algorithms' reward curves."""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Compare Algorithms")
        dialog.geometry("400x260")
        dialog.focus_force()
        dialog.grab_set()

        # ----- Action Space Selection -----
        ctk.CTkLabel(dialog, text="Select Action Space:", text_color="#ecf0f1",
                     font=("Roboto", 13), anchor="center").pack(pady=5, fill="x")
        action_space_var = ctk.StringVar(value="Discrete")
        action_space_menu = ctk.CTkComboBox(
            dialog, values=["Discrete", "Continuous"], variable=action_space_var,
            width=200, font=("Roboto", 11), dropdown_font=("Roboto", 11),
            fg_color="#ecf0f1", button_color="#4CAF50",
            button_hover_color="#45A049", text_color="#2c3e50",
            state="readonly", justify="center"
        )
        action_space_menu.pack(anchor="center")

        # ----- Environment Selection -----
        ctk.CTkLabel(dialog, text="Select Environment:", text_color="#ecf0f1",
                     font=("Roboto", 13), anchor="center").pack(pady=5, fill="x")
        env_var = ctk.StringVar()
        env_menu = ctk.CTkComboBox(
            dialog, values=[], variable=env_var,
            width=200, font=("Roboto", 11), dropdown_font=("Roboto", 11),
            fg_color="#ecf0f1", button_color="#4CAF50",
            button_hover_color="#45A049", text_color="#2c3e50",
            state="readonly", justify="center"
        )
        env_menu.pack(anchor="center")

        def update_env_menu(event=None):
            action = action_space_var.get()
            envs = self.valid_combinations.get(action, {})
            env_list = list(envs.values())[0] if envs else []
            env_menu.configure(values=env_list)
            env_var.set(env_list[0] if env_list else "")

        action_space_menu.configure(command=update_env_menu)
        update_env_menu()

        # ----- Comparison Plot Generation -----
        def generate_plot():
            action_space = action_space_var.get()
            env_name = env_var.get()
            if not action_space or not env_name:
                messagebox.showerror("Error", "Please select action space and environment.")
                return

            if action_space == "Discrete":
                algorithms = ["DQN", "A2C"]
                base_colors = ["#0072B2", "#D55E00"]
                light_colors = ["#8CB4E2", "#EDB687"]
            else:
                algorithms = ["NAF", "SAC"]
                base_colors = ["#009E73", "#CC79A7"]
                light_colors = ["#A3D5C6", "#E6A8CA"]

            try:
                print(f"Generating reward comparison plot for environment: {env_name}")
                print(f"Selected Action Space: {action_space}")
                print(f"Comparing algorithms: {', '.join(algorithms)}\n")

                fig, ax = plt.subplots(figsize=(10, 6))
                if env_name == "FrozenLake-v1":
                    map_size = int(self.map_size_var.get())
                else:
                    map_size = 8

                hps = get_hyperparameters(algorithms[0], env_name, "train", map_size)
                base_plot_path = os.path.join(
                    hps["plot_path"].rsplit("/", 2)[0],
                    env_name
                )

                for algo, color, light_color in zip(algorithms, base_colors, light_colors):
                    if env_name == "FrozenLake-v1":
                        map_size = int(self.map_size_var.get())
                        map_folder = f"{map_size}x{map_size}"  # e.g., "8x8" or "4x4"
                        base_path = base_plot_path.replace("/FrozenLake-v1", "")
                        csv_path = os.path.join(base_path, algo, map_folder,
                                                f"results_{algo}_{env_name}.csv")
                    else:
                        csv_path = os.path.join(base_plot_path, algo, f"results_{algo}_{env_name}.csv")

                    print(f"Loading data from: {csv_path}")
                    if not os.path.exists(csv_path):
                        raise FileNotFoundError(f"CSV file not found: {csv_path}")

                    df = pd.read_csv(csv_path)
                    episodes = df["Episode"]
                    rewards = df["Reward"]
                    print(
                        f"{algo}: Loaded {len(df)} entries. Reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")

                    ax.plot(episodes, rewards, linestyle='-', alpha=0.1, linewidth=1,
                            label=f"{algo} raw", color=color)

                    ma = rewards.rolling(window=20, min_periods=1).mean()
                    ax.plot(episodes, ma, linestyle='-', alpha=0.9, linewidth=2,
                            label=f"{algo} MA(20)", color=color)

                ax.set_title(f"Reward Comparison on {env_name}", fontweight="bold")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.legend(loc="upper left", fontsize=9)
                ax.grid(True, linestyle="--", alpha=0.5)

                save_dir = os.path.join("plots", env_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"comparison_{env_name}_{action_space}.png")
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✅ Plot saved to: {save_path}")


                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    fig.savefig(tmp.name, dpi=100, bbox_inches="tight")
                    img = Image.open(tmp.name)
                plt.close(fig)

                plot_win = ctk.CTkToplevel(dialog)
                plot_win.title("Comparison Plot")
                plot_win.focus_force()
                plot_win.grab_set()
                photo = ImageTk.PhotoImage(img)
                lbl = ctk.CTkLabel(plot_win, image=photo, text="")
                lbl.image = photo
                lbl.pack(padx=10, pady=10)

                def on_close():
                    plot_win.destroy()
                    os.remove(tmp.name)

                plot_win.protocol("WM_DELETE_WINDOW", on_close)

                print("\nPlot generated and displayed successfully.\n" +
                      "Use this comparison to analyze convergence speed, stability, and reward trends between algorithms.\n")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate plot:\n{e}")
                print(f"Error generating plot: {e}")

        generate_button = ctk.CTkButton(
            dialog, text="Generate Comparison Plot", command=generate_plot,
            fg_color="#1abc9c", hover_color="#16a085",
            font=("Roboto", 12), corner_radius=10, width=220, text_color="white"
        )
        generate_button.pack(pady=15, anchor="center")

    def run_experiment(self):
        """Run the selected RL experiment in a separate thread with stop support."""
        algorithm = self.algorithm_var.get().upper()
        env_name = self.env_var.get()
        mode = self.mode_var.get()
        if not algorithm or not env_name:
            messagebox.showerror("Error", "Please select valid options.")
            return

        if env_name == "FrozenLake-v1":
            map_size = int(self.map_size_var.get())
        else:
            map_size = 8

        hyperparams = get_hyperparameters(algorithm, env_name, mode, map_size)
        if env_name == "FrozenLake-v1":
            hyperparams["map_size"] = int(self.map_size_var.get())
            hyperparams["num_states"] = hyperparams["map_size"] ** 2

        ensure_directories_exist(hyperparams)
        print("Directories ensured:")
        pprint(hyperparams)

        pprint(f"Running experiment with Algorithm: {algorithm}, "
              f"Environment: {env_name}, Mode: {mode}, "
              f"Map Size: {hyperparams.get('map_size', 'N/A')}")

        def run_in_thread():
            try:
                DRL = Model_TrainTest(hyperparams, algorithm, env_name)
                self.run_button.configure(state="disabled")
                DRL.run(max_episodes=hyperparams['max_episodes'])
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success",
                    f"Experiment completed: {algorithm} on {env_name} in {mode} mode"
                ))
                self.run_button.configure(state="normal")
            except Exception as error:
                error_message = str(error)
                self.root.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Failed to run experiment: {error_message}"
                ))
            finally:
                self.root.after(0, lambda: self.run_button.configure(state="normal"))
                self.experiment_thread = None

        self.experiment_thread = threading.Thread(target=run_in_thread, daemon=True)
        self.experiment_thread.start()

    def show_structure(self):
        """Display saved structure images from the plots directory."""
        algorithm = self.algorithm_var.get().upper()
        env_name = self.env_var.get()
        mode = self.mode_var.get()

        if not algorithm or not env_name:
            messagebox.showerror("Error", "Please select valid algorithm and environment.")
            return

        if env_name == "FrozenLake-v1":
            map_size = int(self.map_size_var.get())
        else:
            map_size = 8

        hp = get_hyperparameters(algorithm, env_name, mode, map_size)
        ensure_directories_exist(hp)
        plot_dir = './plots/'
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_files = [f for f in os.listdir(plot_dir) if
                       f.lower().endswith(image_extensions) and 'structure' in f.lower()]

        if not image_files:
            messagebox.showerror("Error", f"No structure images found in {plot_dir}.")
            return

        window = ctk.CTkToplevel()
        window.title(f"Structure")
        window.focus_force()
        window.grab_set()
        window.geometry("800x600")

        canvas = ctk.CTkCanvas(window, highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(window, orientation="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas, fg_color="transparent", corner_radius=10)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas_width = canvas.winfo_width()
            frame_width = scrollable_frame.winfo_reqwidth()
            if canvas_width > frame_width:
                canvas.itemconfig(canvas_frame, anchor="n", width=canvas_width)
            else:
                canvas.itemconfig(canvas_frame, anchor="nw", width=frame_width)

        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="n")

        max_width = 600
        for image_file in image_files:
            image_path = os.path.join(plot_dir, image_file)
            img = Image.open(image_path)
            width, height = img.size
            if width > max_width:
                ratio = max_width / width
                img = img.resize((max_width, int(height * ratio)), Image.Resampling.LANCZOS)
            photo = CTkImage(light_image=img, size=img.size)
            frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent", corner_radius=10)
            frame.pack(pady=10, fill="x", expand=True, anchor="center")

            label = ctk.CTkLabel(
                frame,
                text="Structure",
                text_color="#2c3e50",
                font=("Arial Rounded MT", 12),
                anchor="center"
            )
            label.pack(fill="x")

            image_label = ctk.CTkLabel(frame, image=photo, text="")
            image_label.image = photo
            image_label.pack(anchor="center")

        def on_mouse_wheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        # Bind on enter/leave of the scrollable frame
        scrollable_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_mouse_wheel))
        scrollable_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # Initialize the scroll region
        window.update_idletasks()
        configure_scroll_region(None)

    def show_plots(self):
        """Display saved plot images."""
        algorithm = self.algorithm_var.get().upper()
        env_name = self.env_var.get()
        mode = self.mode_var.get()

        if not algorithm or not env_name:
            messagebox.showerror("Error", "Please select valid algorithm and environment.")
            return

        if env_name == "FrozenLake-v1":
            map_size = int(self.map_size_var.get())
        else:
            map_size = 8

        hp = get_hyperparameters(algorithm, env_name, mode, map_size)
        ensure_directories_exist(hp)
        plot_dir = hp['plot_path']
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_files = [f for f in os.listdir(plot_dir) if f.lower().endswith(image_extensions)]

        if not image_files:
            messagebox.showerror("Error", f"No plot images found in {plot_dir}.")
            return

        window = ctk.CTkToplevel()
        window.title(f"Plots for {algorithm} on {env_name}")
        window.focus_force()
        window.grab_set()
        window.geometry("800x600")

        canvas = ctk.CTkCanvas(window, highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(window, orientation="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas, fg_color="transparent", corner_radius=10)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas_width = canvas.winfo_width()
            frame_width = scrollable_frame.winfo_reqwidth()
            if canvas_width > frame_width:
                canvas.itemconfig(canvas_frame, anchor="n", width=canvas_width)
            else:
                canvas.itemconfig(canvas_frame, anchor="nw", width=frame_width)

        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="n")

        max_width = 600
        for image_file in image_files:
            image_path = os.path.join(plot_dir, image_file)
            img = Image.open(image_path)
            width, height = img.size
            if width > max_width:
                ratio = max_width / width
                img = img.resize((max_width, int(height * ratio)), Image.Resampling.LANCZOS)
            photo = CTkImage(light_image=img, size=img.size)
            frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent", corner_radius=10)
            frame.pack(pady=10, fill="x", expand=True, anchor="center")
            label = ctk.CTkLabel(frame, text=f"{image_file} ({algorithm} on {env_name})", text_color="#2c3e50", font=("Arial Rounded MT", 12), anchor="center")
            label.pack(fill="x")
            image_label = ctk.CTkLabel(frame, image=photo, text="")
            image_label.image = photo
            image_label.pack(anchor="center")

        def on_mouse_wheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        # Bind on enter/leave of the scrollable frame
        scrollable_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_mouse_wheel))
        scrollable_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # Initialize the scroll region
        window.update_idletasks()
        configure_scroll_region(None)

    def show_videos(self):
        """Display video information and play videos."""
        algorithm = self.algorithm_var.get().upper()
        env_name = self.env_var.get()
        if not algorithm or not env_name:
            messagebox.showerror("Error", "Please select valid algorithm and environment.")
            return

        if env_name == "FrozenLake-v1":
            map_size = int(self.map_size_var.get())
        else:
            map_size = 8

        hp = get_hyperparameters(algorithm, env_name, self.mode_var.get(), map_size)
        ensure_directories_exist(hp)
        video_files = [f for f in os.listdir(hp['video_path']) if f.endswith((".mp4", ".avi", ".mkv", ".mov"))]

        if not video_files:
            messagebox.showerror("Error", f"No videos found for {algorithm} on {env_name}.")
            return

        render_fps = hp.get('render_fps', 30)
        if env_name == "HalfCheetah-v4":
            render_fps = 50

        video_info = []
        total_duration = 0
        for video_file in video_files:
            video_path = os.path.join(hp['video_path'], video_file)
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or render_fps
                duration = frame_count / fps if fps > 0 else 0
                total_duration += duration
                video_info.append(f"Video: {video_file}, FPS: {fps:.2f}, Duration: {duration:.2f} seconds")
                cap.release()
            else:
                video_info.append(f"Video: {video_file}, Error: Failed to open")

        info_message = f"Algorithm: {algorithm}\nEnvironment: {env_name}\nRender FPS: {render_fps}\n"
        info_message += f"Total Duration: {total_duration:.2f} seconds\n\n"
        info_message += "\n".join(video_info)
        messagebox.showinfo("Video Information", info_message)

        print("Video Information:")
        print(f"Algorithm: {algorithm}")
        print(f"Environment: {env_name}")
        print(f"Render FPS: {render_fps}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        for info in video_info:
            print(info)

        video_window = ctk.CTkToplevel()
        video_window.title(f"Videos for {algorithm} on {env_name}")
        video_window.focus_force()
        video_window.grab_set()
        video_label = ctk.CTkLabel(video_window, text="")
        video_label.pack()
        frame_delay = int(1000 / render_fps)

        def play_video(video_index=0):
            if video_index >= len(video_files):
                video_window.destroy()
                return
            video_path = os.path.join(hp['video_path'], video_files[video_index])
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Failed to open video: {video_path}")
                video_window.destroy()
                return

            def update_frame():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (500, 500))
                    image = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image)
                    video_label.configure(image=photo)
                    video_label.image = photo
                    video_window.after(frame_delay, update_frame)
                else:
                    cap.release()
                    play_video(video_index + 1)

            update_frame()

        try:
            play_video()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play videos: {str(e)}")
            video_window.destroy()

if __name__ == '__main__':
    root = ctk.CTk()
    app = RLExperimentGUI(root)
    root.mainloop()