#!/usr/bin/env python3
"""
SystemC Poker Solver GUI

A Python-based GUI for the SystemC poker model using Tkinter.
Provides an intuitive interface for poker analysis with TOOL_TAGS.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
import sys
from typing import Optional, Dict, List

# Import the existing SystemC runner
from systemC_runner import SystemCRunner, DEPENDENCIES_AVAILABLE

class PokerSolverGUI:
    """Main GUI class for the poker solver."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.runner: Optional[SystemCRunner] = None
        self.model_loaded = False
        
        # Poker data
        self.ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        self.suits = ['♠', '♥', '♦', '♣']
        self.suit_symbols = {'♠': 's', '♥': 'h', '♦': 'd', '♣': 'c'}
        self.positions = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
        self.villain_profiles = [
            "balanced",
            "tag (tight-aggressive)", 
            "lag (loose-aggressive)",
            "nit (super-tight)",
            "station (calling-station)",
            "maniac (loose, hyper-aggressive)",
            "whale (loose-passive)"
        ]
        
        # GUI variables
        self.selected_hero_cards = []
        self.selected_board_cards = []
        self.preflop_actions = [""] * 8  # 8 positions: UTG, UTG+1, LJ, HJ, CO, BTN, SB, BB
        self.preflop_aggressor = None  # Index of first raiser
        self.preflop_aggressor_group = None  # Group of first raiser (EP/MP/LP/BLINDS)
        self.betting_round = 1  # 1=first action, 2=after raise, 3=after 3bet, etc.
        self.action_to_act = 0  # Which position should act next
        self.response_columns = []  # Track additional response columns for 3bets/4bets
        
        # Postflop action state
        self.postflop_actions = ["", ""]  # [OOP, IP] actions
        self.postflop_response_columns = []  # Track postflop response columns
        
        # Auto-calculated values
        self.calculated_pot = 0.0
        self.calculated_stacks = 0.0
        
        # Preflop button storage
        self.preflop_buttons = [[] for _ in range(8)]  # One list per position
        self.preflop_button_frames = []  # Frames for each position
        self.response_button_frames = []  # Frames for response columns
        
        # Postflop button storage  
        self.postflop_buttons = [[] for _ in range(2)]  # [OOP, IP]
        self.postflop_button_frames = []  # Frames for each player
        
        # Model state
        self.loading_model = False
        
        self.setup_gui()
        # Model loading moved to run() method to avoid duplicate calls
        
    def setup_gui(self):
        """Setup the main GUI."""
        self.root.title("SystemC Poker Solver")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2c3e50', foreground='white')
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'), background='#2c3e50', foreground='white')
        style.configure('Card.TButton', font=('Arial', 8, 'bold'))
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🃏 SystemC Poker Solver", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Setup tab
        setup_frame = ttk.Frame(notebook)
        notebook.add(setup_frame, text="Setup Scenario")
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Analysis Results")
        
        self.setup_scenario_tab(setup_frame)
        self.setup_results_tab(results_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Click 'Load Model' to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    def setup_scenario_tab(self, parent):
        """Setup the scenario input tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg='#34495e')
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Model status (above preflop)
        status_frame = ttk.Frame(scrollable_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(status_frame, text="Model Status:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.model_status = ttk.Label(status_frame, text="Loading...", foreground="orange")
        self.model_status.pack(side=tk.LEFT, padx=10)
        
        # Model will auto-load after GUI setup
        
        # Preflop action section (moved to top)
        self.setup_preflop_section(scrollable_frame)
        
        # Cards section with compact boxes
        cards_frame = ttk.LabelFrame(scrollable_frame, text="Cards", padding=10)
        cards_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.setup_compact_card_selectors(cards_frame)
        
        # Continue with the rest of the tab setup...
        self.setup_meta_section(scrollable_frame)
        self.setup_remaining_action_section(scrollable_frame)
        
    def setup_preflop_section(self, parent):
        """Setup preflop action section with multiple buttons per position."""
        self.preflop_frame = ttk.LabelFrame(parent, text="Preflop Action", padding=10)
        self.preflop_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Set minimum height to accommodate 3 buttons per position
        self.preflop_frame.configure(height=120)
        
        self.positions = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
        
        # Action buttons frame
        self.action_frame = ttk.Frame(self.preflop_frame)
        self.action_frame.pack(fill=tk.X, pady=5)
        
        # Position groups for preflop logic
        self.position_groups = {
            'EP': [0, 1],    # UTG, UTG+1
            'MP': [2, 3],    # LJ, HJ 
            'LP': [4, 5],    # CO, BTN
            'BLINDS': [6, 7] # SB, BB
        }
        
        # Initialize preflop state
        self.preflop_actions = [""] * 8  # Current action for each position (empty = not acted)
        self.preflop_button_frames = []  # Frame for each position's buttons
        self.preflop_buttons = []  # Button widgets for each position
        self.preflop_labels = []  # Position name labels
        self.action_to_act = 0  # Current position to act
        self.preflop_aggressor = None  # Position of first raiser
        self.preflop_aggressor_group = None  # Group of first raiser
        self.betting_round = 1  # 1=initial, 2=3bet, 3=4bet, etc.
        
        # Create frames for each position's buttons with labels
        for i, pos in enumerate(self.positions):
            pos_action_frame = ttk.Frame(self.action_frame)
            pos_action_frame.pack(side=tk.LEFT, padx=5)
            
            # Position label inside the button area - wider and black text
            pos_label = tk.Label(pos_action_frame, text=pos, font=('Arial', 9, 'bold'), 
                               width=10, anchor='center', bg='lightgray', fg='black',
                               relief='solid', borderwidth=1)
            pos_label.pack()
            
            self.preflop_button_frames.append(pos_action_frame)
            self.preflop_buttons.append([])
            self.preflop_labels.append(pos_label)
        
        # Create initial button sets
        self.create_preflop_buttons()
        
    def create_preflop_buttons(self):
        """Create initial preflop buttons for all positions."""
        for i in range(8):
            self.update_position_buttons(i)
    
    def update_position_buttons(self, position_index):
        """Update buttons for a specific position based on available actions."""
        current_action = self.preflop_actions[position_index]
        
        # Always clear existing buttons and recreate them to avoid issues
        for btn in self.preflop_buttons[position_index]:
            btn.destroy()
        self.preflop_buttons[position_index].clear()
        
        # Get available actions for this position
        available_actions = self.get_available_actions(position_index)
        
        # Determine which buttons to show based on game state
        if self.preflop_aggressor is None:
            # No one has raised yet - show 2 buttons
            actions_to_show = ["Fold", "Raise 2.5"]
        else:
            # Someone has raised - show 3 buttons
            if position_index in [6, 7]:  # SB, BB
                actions_to_show = ["Fold", "Call", "Raise 11"]
            else:
                actions_to_show = ["Fold", "Call", "Raise 7.5"]
        
        # Create buttons for all actions
        for i, action in enumerate(actions_to_show):
            # Check if action is available and selected
            is_available = action in available_actions
            is_selected = action == current_action
            
            # Determine button styling - FIXED LOGIC
            if is_selected:
                # Selected action: WHITE background, black text, NORMAL state
                bg_color = 'white'
                fg_color = 'black'
                state = tk.NORMAL
            elif not current_action and is_available:
                # Position hasn't acted yet, and this action is available: WHITE
                bg_color = 'white'
                fg_color = 'black'
                state = tk.NORMAL
            else:
                # Action not available or position has acted but this isn't selected: GREY
                bg_color = '#f0f0f0'  # Light grey when disabled
                fg_color = '#888888'  # Grey text for disabled
                state = tk.DISABLED
            
            btn = tk.Button(self.preflop_button_frames[position_index], 
                          text=action,
                          font=('Arial', 8, 'bold'), 
                          width=10, height=1,  # Match label width
                          bg=bg_color, fg=fg_color,
                          state=state,
                          relief='solid',  # Flat box style
                          borderwidth=1,
                          command=lambda pos=position_index, act=action: self.select_preflop_action(pos, act))
            btn.pack(pady=1)
            self.preflop_buttons[position_index].append(btn)
    
    def select_preflop_action(self, position_index, action):
        """Handle preflop action selection with proper sequential logic."""
        positions = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
        
        # If clicking on a position that already has acted, clear all actions to the right
        if self.preflop_actions[position_index]:
            # Clear all actions to the right of this position
            for i in range(position_index + 1, 8):
                self.preflop_actions[i] = ""
            # Reset aggressor if necessary
            if self.preflop_aggressor is not None and self.preflop_aggressor > position_index:
                self.preflop_aggressor = None
                self.preflop_aggressor_group = None
                self.betting_round = 1
            # Clear any response columns since we're resetting
            self.clear_response_columns()
        
        # Auto-assign "fold" to all previous positions that haven't acted
        for i in range(position_index):
            if not self.preflop_actions[i]:
                self.preflop_actions[i] = "Fold"
        
        # Set the action for this position
        self.preflop_actions[position_index] = action
        
        # If this is a raise and we don't have an aggressor yet, set it
        if "Raise" in action and self.preflop_aggressor is None:
            self.preflop_aggressor = position_index
            # Determine aggressor group
            for group, indices in self.position_groups.items():
                if position_index in indices:
                    self.preflop_aggressor_group = group
                    break
            self.betting_round = 2  # Now we're in 3bet territory
        
        # Handle 3bet situation: if someone raises after the initial raiser
        elif "Raise" in action and self.preflop_aggressor is not None and position_index != self.preflop_aggressor:
            # This is a 3bet! Original raiser needs to respond
            self.betting_round = 3  # Now we're in 4bet territory
            
            # Add response column for the original raiser instead of clearing their action
            self.add_response_column(self.preflop_aggressor, "3bet_response")
        
        # Two-player auto-fold logic: if exactly 2 players are active, fold everyone else
        active_players = []
        for i, act in enumerate(self.preflop_actions):
            if act and act != "Fold":
                active_players.append(i)
        
        if len(active_players) == 2:
            # Auto-fold all positions that haven't acted yet
            for i in range(8):
                if not self.preflop_actions[i]:  # Haven't acted yet
                    self.preflop_actions[i] = "Fold"
        
        # Update which position can act next
        self.action_to_act = position_index + 1
        
        # Update all button displays
        self.update_preflop_display()
    
    def get_available_actions(self, position_index):
        """Get available actions for a position based on preflop logic."""
        # Position groups
        EP = [0, 1]  # UTG, UTG+1
        MP = [2, 3]  # LJ, HJ
        LP = [4, 5]  # CO, BTN
        BLINDS = [6, 7]  # SB, BB
        
        # If position already acted, they can only do what they did
        if self.preflop_actions[position_index]:
            return [self.preflop_actions[position_index]]
        
        # If no one has raised yet, only fold and raise available
        if self.preflop_aggressor is None:
            return ["Fold", "Raise 2.5"]
        
        # Someone has raised - check for special cases
        
        # Check if this is the original raiser responding to a 3bet
        three_bettor = None
        for i, action in enumerate(self.preflop_actions):
            if action and "Raise" in action and i != self.preflop_aggressor:
                three_bettor = i
                break
        
        if three_bettor is not None and position_index == self.preflop_aggressor:
            # Original raiser responding to 3bet - can fold, call, or 4bet
            return ["Fold", "Call", "Raise 22"]
        
        # Check if this position is in the same group as aggressor
        current_group = None
        aggressor_group = self.preflop_aggressor_group
        
        for group, indices in self.position_groups.items():
            if position_index in indices:
                current_group = group
                break
        
        # Same group restriction: if someone in your group raised first, you can only fold
        if (current_group == aggressor_group and 
            position_index != self.preflop_aggressor and 
            position_index > self.preflop_aggressor):
            return ["Fold"]
        
        # Normal post-raise options
        if position_index in BLINDS:
            return ["Fold", "Call", "Raise 11"]
        else:
            return ["Fold", "Call", "Raise 7.5"]
    
    def update_preflop_display(self):
        """Update the preflop button display based on current state."""
        # Update all position buttons
        for i in range(8):
            self.update_position_buttons(i)
        
        # Update hero position options to only participating players
        self.update_hero_position_options()
        
        # Update postflop section when preflop changes
        if hasattr(self, 'postflop_frame'):
            # Clear existing postflop UI and recreate
            for widget in self.postflop_action_frame.winfo_children():
                widget.destroy()
            
            self.postflop_button_frames.clear()
            self.postflop_buttons.clear()
            self.postflop_labels.clear()
            self.postflop_actions = ["", ""]
            self.clear_postflop_response_columns()
            
            # Recreate postflop UI with updated players
            participating_players = self.get_participating_players()
            
            if len(participating_players) >= 2:
                # Create frames for OOP and IP players
                for i, (pos_name, pos_index) in enumerate(participating_players[:2]):
                    player_frame = ttk.Frame(self.postflop_action_frame)
                    player_frame.pack(side=tk.LEFT, padx=5)
                    
                    # Player label (OOP/IP with position)
                    player_type = "OOP" if i == 0 else "IP"
                    label_text = f"{player_type}\n({pos_name})"
                    
                    player_label = tk.Label(player_frame, text=label_text, font=('Arial', 9, 'bold'), 
                                           width=10, anchor='center', bg='lightgray', fg='black',
                                           relief='solid', borderwidth=1)
                    player_label.pack()
                    
                    self.postflop_button_frames.append(player_frame)
                    self.postflop_buttons.append([])
                    self.postflop_labels.append(player_label)
                
                # Create initial postflop buttons
                self.create_postflop_buttons()
            else:
                # Not enough players for postflop
                no_action_label = ttk.Label(self.postflop_action_frame, 
                                           text="Select at least 2 players in preflop to enable postflop actions",
                                           font=('Arial', 10))
                no_action_label.pack(pady=20)
        
        # Calculate pot and stacks
        self.calculate_pot_and_stacks()
        
        # Update the title text to show current preflop action sequence
        action_sequence = []
        for i, action in enumerate(self.preflop_actions):
            if action:  # If position has acted
                action_sequence.append(f"{self.positions[i]} {action.lower()}")
        
        # Add response actions to the sequence
        for response in self.response_columns:
            if response['action']:
                pos_name = self.positions[response['position']]
                action_sequence.append(f"{pos_name} {response['action'].lower()}")
        
        if action_sequence:
            title_text = f"Preflop Action: {', '.join(action_sequence)}"
        else:
            title_text = "Preflop Action"
        
        self.preflop_frame.config(text=title_text)
    
    def update_hero_position_options(self):
        """Update hero position dropdown to only show participating players."""
        if hasattr(self, 'pos_combo'):
            # Get participating players
            participating = []
            for i, action in enumerate(self.preflop_actions):
                if action and action != "Fold":
                    participating.append(self.positions[i])
            
            # If no one has acted yet or less than 2 players, allow any position
            if len(participating) < 2:
                available_positions = self.positions
            else:
                available_positions = participating
            
            # Update combobox values
            self.pos_combo['values'] = available_positions
            
            # If current selection is not in available positions, reset to first available
            current_pos = self.hero_pos_var.get()
            if current_pos not in available_positions:
                self.hero_pos_var.set(available_positions[0])
    
    def setup_compact_card_selectors(self, parent):
        """Setup compact card selector boxes."""
        # Hero hand selector
        hero_frame = ttk.Frame(parent)
        hero_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(hero_frame, text="Hero Hand:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.hero_card_btn = tk.Button(hero_frame, text="Click to select cards", 
                                      bg='lightgray', fg='black', font=('Arial', 10),
                                      width=20, height=2,
                                      relief='solid', borderwidth=1,
                                      command=lambda: self.open_card_selector("hero"))
        self.hero_card_btn.pack(side=tk.LEFT, padx=5)
        
        # Board cards selector
        board_frame = ttk.Frame(parent)
        board_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(board_frame, text="Board (Flop):", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.board_card_btn = tk.Button(board_frame, text="Click to select flop", 
                                       bg='lightgray', fg='black', font=('Arial', 10),
                                       width=20, height=2,
                                       relief='solid', borderwidth=1,
                                       command=lambda: self.open_card_selector("board"))
        self.board_card_btn.pack(side=tk.LEFT, padx=5)
    
    def open_card_selector(self, card_type):
        """Open card selection popup."""
        popup = tk.Toplevel(self.root)
        popup.title(f"Select {'Hero Hand' if card_type == 'hero' else 'Board Cards'}")
        popup.geometry("900x300")  # Made wider to fit all 13 ranks
        popup.configure(bg='#2c3e50')
        popup.resizable(False, False)
        
        # Make popup modal
        popup.transient(self.root)
        popup.grab_set()
        
        # Center the popup
        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")
        
        # Selected cards display
        selected_frame = ttk.Frame(popup)
        selected_frame.pack(pady=10)
        
        if card_type == "hero":
            selected_label = tk.Label(selected_frame, text=f"Selected: {', '.join(self.selected_hero_cards) if self.selected_hero_cards else 'None'}", 
                                    font=('Arial', 12, 'bold'), bg='#2c3e50', fg='white')
            max_cards = 2
        else:
            selected_label = tk.Label(selected_frame, text=f"Selected: {', '.join(self.selected_board_cards) if self.selected_board_cards else 'None'}", 
                                    font=('Arial', 12, 'bold'), bg='#2c3e50', fg='white')
            max_cards = 3
        
        selected_label.pack()
        
        # Card grid
        card_grid_frame = ttk.Frame(popup)
        card_grid_frame.pack(pady=20, padx=20)
        
        # Create card buttons organized by suits
        for suit_idx, suit in enumerate(self.suits):
            suit_frame = ttk.Frame(card_grid_frame)
            suit_frame.pack(pady=2)
            
            # Suit label
            suit_color = 'red' if suit in ['♥', '♦'] else 'black'
            suit_label = tk.Label(suit_frame, text=suit, font=('Arial', 16, 'bold'), 
                                 fg=suit_color, bg='#2c3e50', width=2)
            suit_label.pack(side=tk.LEFT, padx=5)
            
            # Rank buttons for this suit
            for rank in self.ranks:
                card_text = f"{rank}{suit}"
                
                # Check if card is already selected in this category
                is_selected_here = card_text in (self.selected_hero_cards if card_type == "hero" else self.selected_board_cards)
                
                # Check if card is used in the OTHER category (CARD OVERLAP PREVENTION)
                is_used_elsewhere = card_text in (self.selected_board_cards if card_type == "hero" else self.selected_hero_cards)
                
                # Determine button state and color
                if is_used_elsewhere:
                    # Card is used elsewhere - GREY and DISABLED
                    bg_color = '#f0f0f0'
                    fg_color = '#888888'
                    state = tk.DISABLED
                    command = None
                elif is_selected_here:
                    # Card is selected in this category - LIGHT BLUE
                    bg_color = 'lightblue'
                    fg_color = suit_color
                    state = tk.NORMAL
                    command = lambda c=card_text: self.toggle_card_in_popup(c, card_type, selected_label, popup)
                else:
                    # Card is available - WHITE
                    bg_color = 'white'
                    fg_color = suit_color
                    state = tk.NORMAL
                    command = lambda c=card_text: self.toggle_card_in_popup(c, card_type, selected_label, popup)
                
                btn = tk.Button(suit_frame, text=card_text, font=('Arial', 9, 'bold'),
                               width=3, height=2,
                               bg=bg_color, fg=fg_color, state=state,
                               command=command)
                btn.pack(side=tk.LEFT, padx=1)
        
        # Control buttons
        btn_frame = ttk.Frame(popup)
        btn_frame.pack(pady=20)
        
        # Clear button
        clear_btn = tk.Button(btn_frame, text="Clear", font=('Arial', 11), 
                             width=12, height=2,
                             bg='lightcoral', fg='black',
                             command=lambda: self.clear_cards_in_popup(card_type, selected_label, popup))
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Save button - more prominent
        save_btn = tk.Button(btn_frame, text="💾 SAVE CARDS", font=('Arial', 11, 'bold'), 
                            width=15, height=2,
                            bg='lightgreen', fg='black',
                            command=lambda: self.confirm_card_selection(card_type, popup))
        save_btn.pack(side=tk.LEFT, padx=10)
        
        # Cancel button
        cancel_btn = tk.Button(btn_frame, text="Cancel", font=('Arial', 11), 
                              width=12, height=2,
                              bg='lightgray', fg='black',
                              command=popup.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=10)
    
    def toggle_card_in_popup(self, card, card_type, selected_label, popup):
        """Toggle card selection in popup."""
        # Don't allow selection if card is used in the other category
        other_cards = self.selected_board_cards if card_type == "hero" else self.selected_hero_cards
        if card in other_cards:
            return  # Card is blocked, do nothing
        
        if card_type == "hero":
            if card in self.selected_hero_cards:
                self.selected_hero_cards.remove(card)
            elif len(self.selected_hero_cards) < 2:
                self.selected_hero_cards.append(card)
            selected_label.config(text=f"Selected: {', '.join(self.selected_hero_cards) if self.selected_hero_cards else 'None'}")
        else:
            if card in self.selected_board_cards:
                self.selected_board_cards.remove(card)
            elif len(self.selected_board_cards) < 3:
                self.selected_board_cards.append(card)
            selected_label.config(text=f"Selected: {', '.join(self.selected_board_cards) if self.selected_board_cards else 'None'}")
        
        # Update ALL button colors in popup to reflect current state
        self.refresh_popup_buttons(popup, card_type)
    
    def refresh_popup_buttons(self, popup, card_type):
        """Refresh all button colors in the card selection popup."""
        # Navigate through the popup widget hierarchy to find card buttons
        for widget in popup.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, ttk.Frame):
                                # This is a suit frame
                                for button in grandchild.winfo_children():
                                    if isinstance(button, tk.Button) and hasattr(button, 'cget'):
                                        btn_text = button.cget('text')
                                        if len(btn_text) == 2:  # This is a card button
                                            # Check selection states
                                            is_selected_here = btn_text in (self.selected_hero_cards if card_type == "hero" else self.selected_board_cards)
                                            is_used_elsewhere = btn_text in (self.selected_board_cards if card_type == "hero" else self.selected_hero_cards)
                                            
                                            # Get suit color
                                            suit = btn_text[1]
                                            suit_color = 'red' if suit in ['♥', '♦'] else 'black'
                                            
                                            # Apply appropriate colors
                                            if is_used_elsewhere:
                                                # Card is used elsewhere - GREY and DISABLED
                                                button.config(bg='#f0f0f0', fg='#888888', state=tk.DISABLED)
                                            elif is_selected_here:
                                                # Card is selected in this category - LIGHT BLUE
                                                button.config(bg='lightblue', fg=suit_color, state=tk.NORMAL)
                                            else:
                                                # Card is available - WHITE
                                                button.config(bg='white', fg=suit_color, state=tk.NORMAL)
    
    def clear_cards_in_popup(self, card_type, selected_label, popup):
        """Clear cards in popup."""
        if card_type == "hero":
            self.selected_hero_cards.clear()
            selected_label.config(text="Selected: None")
        else:
            self.selected_board_cards.clear()
            selected_label.config(text="Selected: None")
        
        # Update button colors using the new refresh function
        self.refresh_popup_buttons(popup, card_type)
    
    def confirm_card_selection(self, card_type, popup):
        """Confirm card selection and update main display."""
        self.update_card_displays()
        popup.destroy()
    
    def update_card_displays(self):
        """Update the compact card display buttons."""
        # Update hero cards
        if self.selected_hero_cards:
            display_cards = []
            for card in self.selected_hero_cards:
                rank = card[0]
                suit = self.suit_symbols[card[1]]
                display_cards.append(f"{rank}{suit}")
            self.hero_card_btn.config(text=f"Hero: {' '.join(display_cards)}", bg='lightgreen')
        else:
            self.hero_card_btn.config(text="Click to select cards", bg='lightgray')
        
        # Update board cards
        if self.selected_board_cards:
            display_cards = []
            for card in self.selected_board_cards:
                rank = card[0]
                suit = self.suit_symbols[card[1]]
                display_cards.append(f"{rank}{suit}")
            self.board_card_btn.config(text=f"Flop: {' '.join(display_cards)}", bg='lightgreen')
        else:
            self.board_card_btn.config(text="Click to select flop", bg='lightgray')
    
    def setup_remaining_action_section(self, parent):
        """Setup postflop action section with auto-calculated values."""
        # Auto-calculated pot and stacks display
        calc_frame = ttk.LabelFrame(parent, text="Auto-Calculated Values", padding=10)
        calc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        calc_info_frame = ttk.Frame(calc_frame)
        calc_info_frame.pack(fill=tk.X, pady=5)
        
        # Pot size (auto-calculated)
        ttk.Label(calc_info_frame, text="Pot Size (bb):", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.pot_display = ttk.Label(calc_info_frame, text="0.0", font=('Arial', 10))
        self.pot_display.pack(side=tk.LEFT, padx=(5, 20))
        
        # Effective stacks (auto-calculated)
        ttk.Label(calc_info_frame, text="Effective Stacks (bb):", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.stacks_display = ttk.Label(calc_info_frame, text="100.0", font=('Arial', 10))
        self.stacks_display.pack(side=tk.LEFT, padx=5)
        
        # Postflop action section
        self.setup_postflop_section(parent)
        
        # Analyze button
        analyze_frame = ttk.Frame(parent)
        analyze_frame.pack(fill=tk.X, padx=10, pady=20)
        
        self.analyze_btn = ttk.Button(analyze_frame, text="🚀 Analyze Hand", 
                                     command=self.analyze_hand_thread, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(analyze_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
    
    def setup_postflop_section(self, parent):
        """Setup postflop action section similar to preflop."""
        self.postflop_frame = ttk.LabelFrame(parent, text="Postflop Action", padding=10)
        self.postflop_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Set minimum height
        self.postflop_frame.configure(height=120)
        
        # Action buttons frame
        self.postflop_action_frame = ttk.Frame(self.postflop_frame)
        self.postflop_action_frame.pack(fill=tk.X, pady=5)
        
        # Initialize postflop state
        self.postflop_actions = ["", ""]  # [OOP, IP]
        self.postflop_button_frames = []
        self.postflop_buttons = []
        self.postflop_labels = []
        self.postflop_response_columns = []
        
        # Get participating players for postflop
        participating_players = self.get_participating_players()
        
        if len(participating_players) >= 2:
            # Create frames for OOP and IP players
            for i, (pos_name, pos_index) in enumerate(participating_players[:2]):
                player_frame = ttk.Frame(self.postflop_action_frame)
                player_frame.pack(side=tk.LEFT, padx=5)
                
                # Player label (OOP/IP with position)
                player_type = "OOP" if i == 0 else "IP"
                label_text = f"{player_type}\n({pos_name})"
                
                player_label = tk.Label(player_frame, text=label_text, font=('Arial', 9, 'bold'), 
                                       width=10, anchor='center', bg='lightgray', fg='black',
                                       relief='solid', borderwidth=1)
                player_label.pack()
                
                self.postflop_button_frames.append(player_frame)
                self.postflop_buttons.append([])
                self.postflop_labels.append(player_label)
            
            # Create initial postflop buttons
            self.create_postflop_buttons()
        else:
            # Not enough players for postflop
            no_action_label = ttk.Label(self.postflop_action_frame, 
                                       text="Select at least 2 players in preflop to enable postflop actions",
                                       font=('Arial', 10))
            no_action_label.pack(pady=20)
    
    def get_participating_players(self):
        """Get list of (position_name, position_index) for non-folded players in POSTFLOP ACTION ORDER."""
        participating = []
        
        # Collect all non-folded players
        for i, action in enumerate(self.preflop_actions):
            if action and action != "Fold":
                participating.append((self.positions[i], i))
        
        # Add response column players
        for response in self.response_columns:
            if response['action'] and response['action'] != "Fold":
                pos_name = self.positions[response['position']]
                # Only add if not already in list
                if not any(p[1] == response['position'] for p in participating):
                    participating.append((pos_name, response['position']))
        
        # Sort by postflop action order: SB(6), BB(7), UTG(0), UTG+1(1), LJ(2), HJ(3), CO(4), BTN(5)
        postflop_order = [6, 7, 0, 1, 2, 3, 4, 5]  # SB, BB, UTG, UTG+1, LJ, HJ, CO, BTN
        
        def postflop_sort_key(player_tuple):
            pos_index = player_tuple[1]
            try:
                return postflop_order.index(pos_index)
            except ValueError:
                return 999  # Should not happen
        
        participating.sort(key=postflop_sort_key)
        return participating
    
    def create_postflop_buttons(self):
        """Create initial postflop buttons for both players."""
        for i in range(min(2, len(self.postflop_button_frames))):
            self.update_postflop_buttons(i)
    
    def update_postflop_buttons(self, player_index):
        """Update buttons for a specific postflop player."""
        if player_index >= len(self.postflop_button_frames):
            return
            
        current_action = self.postflop_actions[player_index]
        
        # Clear existing buttons
        for btn in self.postflop_buttons[player_index]:
            btn.destroy()
        self.postflop_buttons[player_index].clear()
        
        # Get available actions
        available_actions = self.get_postflop_available_actions(player_index)
        
        # Create buttons for all actions
        for action in available_actions:
            is_selected = action == current_action
            
            # Button styling - SAME AS PREFLOP
            if is_selected:
                bg_color = 'white'
                fg_color = 'black'
                state = tk.NORMAL
            elif not current_action:
                bg_color = 'white'
                fg_color = 'black'
                state = tk.NORMAL
            else:
                bg_color = '#f0f0f0'
                fg_color = '#888888'
                state = tk.DISABLED
            
            btn = tk.Button(self.postflop_button_frames[player_index], 
                          text=action,
                          font=('Arial', 8, 'bold'), 
                          width=10, height=1,
                          bg=bg_color, fg=fg_color,
                          state=state,
                          relief='solid', borderwidth=1,
                          command=lambda p=player_index, a=action: self.select_postflop_action(p, a))
            btn.pack(pady=1)
            self.postflop_buttons[player_index].append(btn)
    
    def get_postflop_available_actions(self, player_index):
        """Get available postflop actions for a player."""
        if player_index == 0:  # OOP player
            if not self.postflop_actions[0]:  # OOP hasn't acted
                return ["Check", "Bet 33%", "Bet 50%", "Bet 100%", "All-in"]
            else:
                return [self.postflop_actions[0]]  # Already acted
        else:  # IP player (index 1)
            oop_action = self.postflop_actions[0]
            
            if not oop_action:  # OOP hasn't acted yet
                return []  # IP can't act until OOP acts
            elif oop_action == "Check":
                if not self.postflop_actions[1]:  # IP hasn't acted after check
                    return ["Check", "Bet 33%", "Bet 50%", "Bet 100%", "All-in"]
                else:
                    return [self.postflop_actions[1]]  # Already acted
            elif "Bet" in oop_action or "All-in" in oop_action:
                if not self.postflop_actions[1]:  # IP hasn't responded to bet
                    return ["Fold", "Call", "Raise 75%", "All-in"]
                else:
                    return [self.postflop_actions[1]]  # Already acted
        
        return []
    
    def select_postflop_action(self, player_index, action):
        """Handle postflop action selection."""
        # Set the action
        self.postflop_actions[player_index] = action
        
        # Handle special cases for response columns
        if player_index == 1 and "Bet" in action and self.postflop_actions[0] == "Check":
            # IP bet after OOP check - add response column for OOP
            self.add_postflop_response_column(0, "bet_response")
        
        # Update button displays
        self.update_postflop_display()
    
    def add_postflop_response_column(self, player_index, response_type):
        """Add a response column for postflop betting."""
        participating = self.get_participating_players()
        if player_index >= len(participating):
            return
            
        player_name = participating[player_index][0]
        
        # Check if this response column already exists
        for response in self.postflop_response_columns:
            if response['player'] == player_index and response['type'] == response_type:
                return
        
        # Create new response column frame
        response_frame = ttk.Frame(self.postflop_action_frame)
        response_frame.pack(side=tk.LEFT, padx=5)
        
        # Player label - SAME STYLING AS PREFLOP
        pos_label = tk.Label(response_frame, text=player_name, 
                           font=('Arial', 8, 'bold'), 
                           width=10, height=2,
                           bg='lightblue', fg='black',
                           relief='solid', borderwidth=1)
        pos_label.pack(pady=1)
        
        # Button frame
        button_frame = ttk.Frame(response_frame)
        button_frame.pack()
        
        # Response actions
        response_actions = ["Fold", "Call", "Raise 75%", "All-in"]
        
        # Create buttons - SAME STYLING AS PREFLOP
        response_buttons = []
        for action in response_actions:
            btn = tk.Button(button_frame, 
                          text=action,
                          font=('Arial', 8, 'bold'), 
                          width=10, height=1,
                          bg='white', fg='black',
                          relief='solid', borderwidth=1,
                          command=lambda p=player_index, a=action, t=response_type: 
                                 self.select_postflop_response_action(p, a, t))
            btn.pack(pady=1)
            response_buttons.append(btn)
        
        # Store response column info
        response_info = {
            'player': player_index,
            'type': response_type,
            'frame': response_frame,
            'buttons': response_buttons,
            'action': ""
        }
        self.postflop_response_columns.append(response_info)
    
    def select_postflop_response_action(self, player_index, action, response_type):
        """Handle postflop response action selection."""
        # Find and update the response column
        for response in self.postflop_response_columns:
            if response['player'] == player_index and response['type'] == response_type:
                response['action'] = action
                
                # Update button styling - SAME AS PREFLOP
                for btn in response['buttons']:
                    btn_text = btn.cget('text')
                    if btn_text == action:
                        btn.config(bg='white', fg='black', state=tk.NORMAL)
                    else:
                        btn.config(bg='#f0f0f0', fg='#888888', state=tk.DISABLED)
                break
        
        self.update_postflop_display()
    
    def update_postflop_display(self):
        """Update postflop button displays and calculate pot/stacks."""
        # Update buttons
        for i in range(len(self.postflop_button_frames)):
            self.update_postflop_buttons(i)
        
        # Calculate and update pot/stacks
        self.calculate_pot_and_stacks()
        
        # Update postflop title
        action_sequence = []
        participating = self.get_participating_players()
        
        for i, action in enumerate(self.postflop_actions):
            if action and i < len(participating):
                player_type = "OOP" if i == 0 else "IP"
                action_sequence.append(f"{player_type} {action.lower()}")
        
        # Add response actions
        for response in self.postflop_response_columns:
            if response['action'] and response['player'] < len(participating):
                player_type = "OOP" if response['player'] == 0 else "IP"
                action_sequence.append(f"{player_type} {response['action'].lower()}")
        
        if action_sequence:
            title_text = f"Postflop Action: {', '.join(action_sequence)}"
        else:
            title_text = "Postflop Action"
        
        self.postflop_frame.config(text=title_text)
    
    def calculate_pot_and_stacks(self):
        """Auto-calculate pot size and effective stacks from preflop actions including blinds."""
        starting_stacks = 100.0
        
        # Initialize with blinds
        investments = [0.0] * 8  # Track each position's investment
        investments[6] = 0.5  # SB posts 0.5bb
        investments[7] = 1.0  # BB posts 1.0bb
        
        # Process main preflop actions
        for i, action in enumerate(self.preflop_actions):
            if action and action != "Fold":
                if "Raise" in action:
                    size_str = action.split()[-1] if len(action.split()) > 1 else "2.5"
                    try:
                        size = float(size_str)
                        investments[i] = size  # Total investment (not additional)
                    except ValueError:
                        investments[i] = 2.5  # Default
                elif action == "Call":
                    # Find the current bet size to call
                    current_bet = max(investments) if any(investments) else 1.0
                    investments[i] = current_bet  # Match the current bet
        
        # Process response column actions
        for response in self.response_columns:
            if response['action'] and response['action'] != "Fold":
                pos_index = response['position']
                if "Raise" in response['action']:
                    size_str = response['action'].split()[-1] if len(response['action'].split()) > 1 else "22"
                    try:
                        size = float(size_str)
                        investments[pos_index] = size  # Total investment
                    except ValueError:
                        investments[pos_index] = 22.0  # Default 4bet size
                elif response['action'] == "Call":
                    # Find current bet to call
                    current_bet = max(investments) if any(investments) else 1.0
                    investments[pos_index] = current_bet
        
        # Calculate pot (sum of all investments)
        pot = sum(investments)
        
        # Calculate effective stacks (starting stacks minus investment of participating players)
        participating_investments = []
        for i, action in enumerate(self.preflop_actions):
            if action and action != "Fold":
                participating_investments.append(investments[i])
        
        # Add response column investments
        for response in self.response_columns:
            if response['action'] and response['action'] != "Fold":
                pos_index = response['position']
                participating_investments.append(investments[pos_index])
        
        if participating_investments:
            min_investment = min(participating_investments)
            effective_stacks = starting_stacks - min_investment
        else:
            # Default case with blinds only
            effective_stacks = starting_stacks - 1.0  # BB investment
        
        # Update displays
        self.calculated_pot = pot
        self.calculated_stacks = effective_stacks
        
        if hasattr(self, 'pot_display'):
            self.pot_display.config(text=f"{pot:.1f}")
        if hasattr(self, 'stacks_display'):
            self.stacks_display.config(text=f"{effective_stacks:.1f}")
    
    def clear_postflop_response_columns(self):
        """Clear all postflop response columns."""
        for response in self.postflop_response_columns:
            response['frame'].destroy()
        self.postflop_response_columns.clear()
    
    def setup_card_selector(self, parent, card_type):
        """Setup card selector grid."""
        # Selected cards display
        if card_type == "hero":
            self.hero_cards_label = ttk.Label(parent, text="Selected: None", font=('Arial', 10, 'bold'))
            self.hero_cards_label.pack(pady=5)
            max_cards = 2
        else:
            self.board_cards_label = ttk.Label(parent, text="Selected: None", font=('Arial', 10, 'bold'))
            self.board_cards_label.pack(pady=5)
            max_cards = 3
        
        # Card grid (organized by suit rows)
        card_frame = ttk.Frame(parent)
        card_frame.pack(pady=5)
        
        # Create buttons for each card
        for suit_idx, suit in enumerate(self.suits):
            suit_frame = ttk.Frame(card_frame)
            suit_frame.pack(fill=tk.X, pady=1)
            
            # Suit label
            suit_color = 'red' if suit in ['♥', '♦'] else 'black'
            suit_label = tk.Label(suit_frame, text=suit, font=('Arial', 14, 'bold'), 
                                 fg=suit_color, width=3)
            suit_label.pack(side=tk.LEFT)
            
            # Rank buttons for this suit
            for rank in self.ranks:
                card_text = f"{rank}{suit}"
                btn = tk.Button(suit_frame, text=card_text, font=('Arial', 8, 'bold'),
                               width=4, height=1,
                               command=lambda c=card_text, t=card_type: self.toggle_card(c, t))
                btn.pack(side=tk.LEFT, padx=1)
                
                # Set button color
                if suit in ['♥', '♦']:
                    btn.configure(bg='white', fg='red', activebackground='#ffcccc')
                else:
                    btn.configure(bg='white', fg='black', activebackground='#cccccc')
        
        # Clear button for this section
        ttk.Button(parent, text=f"Clear {card_type.title()}", 
                  command=lambda: self.clear_cards(card_type)).pack(pady=5)
    
    def setup_meta_section(self, parent):
        """Setup position and profile section."""
        # Position and profile
        meta_frame = ttk.LabelFrame(parent, text="Game Information", padding=10)
        meta_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Hero position
        pos_frame = ttk.Frame(meta_frame)
        pos_frame.pack(fill=tk.X, pady=2)
        ttk.Label(pos_frame, text="Hero Position:").pack(side=tk.LEFT)
        self.hero_pos_var = tk.StringVar(value="BTN")
        
        # Get participating players for hero position restriction
        def get_participating_positions():
            """Get positions of players participating in the pot."""
            participating = []
            for i, action in enumerate(self.preflop_actions):
                if action and action != "Fold":
                    participating.append(self.positions[i])
            # If no one has acted yet, allow any position
            return participating if participating else self.positions
        
        self.pos_combo = ttk.Combobox(pos_frame, textvariable=self.hero_pos_var, 
                                     values=get_participating_positions(), state="readonly", width=15)
        self.pos_combo.pack(side=tk.LEFT, padx=10)
        
        # Villain profile
        profile_frame = ttk.Frame(meta_frame)
        profile_frame.pack(fill=tk.X, pady=2)
        ttk.Label(profile_frame, text="Villain Profile:").pack(side=tk.LEFT)
        self.villain_profile_var = tk.StringVar(value="balanced")
        profile_combo = ttk.Combobox(profile_frame, textvariable=self.villain_profile_var,
                                   values=self.villain_profiles, state="readonly", width=25)
        profile_combo.pack(side=tk.LEFT, padx=10)
    
    def setup_action_section(self, parent):
        """Setup action history section."""
        # Action history
        history_frame = ttk.LabelFrame(parent, text="Action History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Preflop history
        ttk.Label(history_frame, text="Preflop History:").pack(anchor=tk.W)
        self.preflop_entry = tk.Text(history_frame, height=3, wrap=tk.WORD)
        self.preflop_entry.pack(fill=tk.X, pady=2)
        self.preflop_entry.insert(1.0, "UTG folds, UTG+1 folds, LJ folds, HJ folds, CO folds, BTN raises 2.5bb, SB folds, BB calls 2.5bb")
        
        # Pot and stacks
        pot_frame = ttk.Frame(history_frame)
        pot_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(pot_frame, text="Pot Size (bb):").pack(side=tk.LEFT)
        self.pot_size_var = tk.StringVar(value="6.5")
        ttk.Entry(pot_frame, textvariable=self.pot_size_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(pot_frame, text="Effective Stacks (bb):").pack(side=tk.LEFT, padx=(20, 5))
        self.stacks_var = tk.StringVar(value="97.5")
        ttk.Entry(pot_frame, textvariable=self.stacks_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Flop actions
        ttk.Label(history_frame, text="Previous Actions (if any):").pack(anchor=tk.W, pady=(10, 0))
        self.actions_entry = tk.Text(history_frame, height=2, wrap=tk.WORD)
        self.actions_entry.pack(fill=tk.X, pady=2)
        
        # Legal actions
        ttk.Label(history_frame, text="Legal Actions:").pack(anchor=tk.W, pady=(10, 0))
        self.legal_actions_var = tk.StringVar(value="[check,bet 33% (2.1bb),bet 50% (3.3bb),bet 100% (6.5bb),allin]")
        ttk.Entry(history_frame, textvariable=self.legal_actions_var).pack(fill=tk.X, pady=2)
        
        # Analyze button
        analyze_frame = ttk.Frame(parent)
        analyze_frame.pack(fill=tk.X, padx=10, pady=20)
        
        self.analyze_btn = ttk.Button(analyze_frame, text="🚀 Analyze Hand", 
                                     command=self.analyze_hand_thread, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(analyze_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
    
    def toggle_card(self, card, card_type):
        """Toggle card selection."""
        if card_type == "hero":
            if card in self.selected_hero_cards:
                self.selected_hero_cards.remove(card)
            elif len(self.selected_hero_cards) < 2:
                self.selected_hero_cards.append(card)
            self.update_hero_display()
        else:  # board
            if card in self.selected_board_cards:
                self.selected_board_cards.remove(card)
            elif len(self.selected_board_cards) < 3:
                self.selected_board_cards.append(card)
            self.update_board_display()
    
    def clear_cards(self, card_type):
        """Clear selected cards."""
        if card_type == "hero":
            self.selected_hero_cards.clear()
            self.update_hero_display()
        else:
            self.selected_board_cards.clear()
            self.update_board_display()
    
    def update_hero_display(self):
        """Update hero cards display."""
        if self.selected_hero_cards:
            # Convert to solver format (As Kh)
            display_cards = []
            for card in self.selected_hero_cards:
                rank = card[0]
                suit = self.suit_symbols[card[1]]
                display_cards.append(f"{rank}{suit}")
            self.hero_cards_label.config(text=f"Selected: {' '.join(display_cards)}")
        else:
            self.hero_cards_label.config(text="Selected: None")
    
    def update_board_display(self):
        """Update board cards display."""
        if self.selected_board_cards:
            # Convert to solver format
            display_cards = []
            for card in self.selected_board_cards:
                rank = card[0]
                suit = self.suit_symbols[card[1]]
                display_cards.append(f"{rank}{suit}")
            self.board_cards_label.config(text=f"Selected: {' '.join(display_cards)}")
        else:
            self.board_cards_label.config(text="Selected: None")
    
    def setup_results_tab(self, parent):
        """Setup the results display tab."""
        # Control buttons at the top
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Generate Prompt button
        self.generate_prompt_btn = tk.Button(control_frame, text="🔧 Generate Prompt", 
                                           font=('Arial', 11, 'bold'),
                                           width=18, height=2,
                                           bg='lightblue', fg='black',
                                           command=self.generate_prompt_thread)
        self.generate_prompt_btn.pack(side=tk.LEFT, padx=5)
        
        # Analyze button (moved to results tab)
        self.analyze_btn_results = tk.Button(control_frame, text="🚀 Analyze Hand", 
                                           font=('Arial', 11, 'bold'),
                                           width=18, height=2,
                                           bg='lightgreen', fg='black',
                                           command=self.analyze_hand_thread, state=tk.DISABLED)
        self.analyze_btn_results.pack(side=tk.LEFT, padx=5)
        
        # Prompt display
        prompt_frame = ttk.LabelFrame(parent, text="Generated Prompt", padding=10)
        prompt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.prompt_display = scrolledtext.ScrolledText(prompt_frame, height=15, wrap=tk.WORD)
        self.prompt_display.pack(fill=tk.BOTH, expand=True)
        
        # Response display
        response_frame = ttk.LabelFrame(parent, text="SystemC Analysis", padding=10)
        response_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.response_display = scrolledtext.ScrolledText(response_frame, height=10, wrap=tk.WORD)
        self.response_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text colors
        self.response_display.tag_configure("action", foreground="white", font=('Arial', 11, 'bold'))
        self.response_display.tag_configure("analysis", foreground="darkgreen")
    
    def load_model_thread(self):
        """Load model in separate thread."""
        def load_model():
            try:
                # Update status on main thread
                self.root.after(0, lambda: self.model_status.config(text="Loading...", foreground="orange"))
                
                # Initialize runner - try multiple possible paths
                possible_paths = [
                    "out/stageB/systemC_bf16_r256",           # If run from project root
                    "../../out/stageB/systemC_bf16_r256",     # If run from scripts/app
                    "../../../out/stageB/systemC_bf16_r256",  # If run from deeper
                ]
                
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if not model_path:
                    raise FileNotFoundError(f"SystemC model not found in any of these locations: {possible_paths}")
                
                # Create runner if it doesn't exist
                if not self.runner:
                    self.runner = SystemCRunner(model_path)
                
                # Load the model
                self.runner.load_model()
                
                # Update GUI on main thread
                def update_success():
                    self.model_loaded = True
                    self.model_status.config(text="✅ Loaded", foreground="green")
                    self.analyze_btn.config(state=tk.NORMAL)
                    if hasattr(self, 'status_var'):
                        self.status_var.set("Model loaded successfully!")
                
                self.root.after(0, update_success)
                
            except Exception as e:
                # Capture error message outside the nested function
                error_msg = f"Failed to load model: {str(e)}"
                
                # Update GUI on main thread
                def update_error():
                    messagebox.showerror("Error", error_msg)
                    self.model_status.config(text="❌ Failed", foreground="red")
                    if hasattr(self, 'status_var'):
                        self.status_var.set("Failed to load model")
                
                self.root.after(0, update_error)
        
        # Run in thread to prevent GUI freezing
        threading.Thread(target=load_model, daemon=True).start()
    
    def analyze_hand_thread(self):
        """Analyze hand in separate thread."""
        def analyze():
            try:
                # Update status on main thread
                if hasattr(self, 'status_var'):
                    self.root.after(0, lambda: self.status_var.set("Analyzing hand..."))
                
                self.root.after(0, lambda: self.analyze_btn.config(state=tk.DISABLED))
                if hasattr(self, 'analyze_btn_results'):
                    self.root.after(0, lambda: self.analyze_btn_results.config(state=tk.DISABLED))
                
                # Check if prompt is already generated
                current_prompt = self.prompt_display.get(1.0, tk.END).strip()
                
                if current_prompt and current_prompt != "":
                    # Use existing prompt
                    prompt = current_prompt
                    if hasattr(self, 'status_var'):
                        self.root.after(0, lambda: self.status_var.set("Using existing prompt for analysis..."))
                else:
                    # Generate new prompt
                    if hasattr(self, 'status_var'):
                        self.root.after(0, lambda: self.status_var.set("Generating prompt and analyzing..."))
                    
                    # Validate inputs
                    if len(self.selected_hero_cards) != 2:
                        raise ValueError("Please select exactly 2 hero cards")
                    
                    if len(self.selected_board_cards) != 3:
                        raise ValueError("Please select exactly 3 board cards")
                    
                    # Build scenario
                    scenario = self.build_scenario()
                    
                    # Generate prompt with TOOL_TAGS (requires existing runner)
                    if not hasattr(self, 'runner') or not self.runner:
                        raise RuntimeError("SystemC runner not initialized! Load model first.")
                    
                    prompt = self.runner.format_prompt(scenario)
                    
                    # Update prompt display on main thread
                    self.root.after(0, lambda: self.prompt_display.delete(1.0, tk.END))
                    self.root.after(0, lambda: self.prompt_display.insert(tk.END, prompt))
                
                # Generate response
                response = self.runner.generate_response(prompt)
                
                # Update response display on main thread
                def update_response():
                    # Cut off response at </s> token if present
                    cleaned_response = response
                    if "</s>" in response:
                        cleaned_response = response.split("</s>")[0].strip()
                    
                    self.response_display.delete(1.0, tk.END)
                    self.response_display.insert(tk.END, cleaned_response)
                    
                    # Highlight action if found
                    if "**Best Action:" in cleaned_response:
                        start = cleaned_response.find("**Best Action:")
                        end = cleaned_response.find("**", start + 2) + 2
                        if end > start:
                            # Calculate text widget positions
                            start_line = cleaned_response[:start].count('\n') + 1
                            start_char = len(cleaned_response[:start].split('\n')[-1])
                            end_line = cleaned_response[:end].count('\n') + 1
                            end_char = len(cleaned_response[:end].split('\n')[-1])
                            
                            self.response_display.tag_add("action", f"{start_line}.{start_char}", f"{end_line}.{end_char}")
                    
                    if hasattr(self, 'status_var'):
                        self.status_var.set("Analysis complete!")
                    
                    # Switch to results tab
                    try:
                        notebook = self.root.winfo_children()[0].winfo_children()[1]  # Get notebook
                        notebook.select(1)  # Select results tab
                    except:
                        pass  # Ignore if notebook structure changed
                
                self.root.after(0, update_response)
                
            except Exception as e:
                # Update GUI on main thread
                error_msg = str(e)  # Capture error message
                def update_error():
                    messagebox.showerror("Error", error_msg)
                    if hasattr(self, 'status_var'):
                        self.status_var.set("Analysis failed")
                
                self.root.after(0, update_error)
            finally:
                # Re-enable buttons on main thread
                if self.model_loaded:
                    self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
                    if hasattr(self, 'analyze_btn_results'):
                        self.root.after(0, lambda: self.analyze_btn_results.config(state=tk.NORMAL))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def generate_prompt_thread(self):
        """Generate prompt in separate thread without running analysis."""
        def generate():
            try:
                # Update status on main thread
                if hasattr(self, 'status_var'):
                    self.root.after(0, lambda: self.status_var.set("Generating prompt..."))
                
                self.root.after(0, lambda: self.generate_prompt_btn.config(state=tk.DISABLED))
                
                # Validate inputs
                if len(self.selected_hero_cards) != 2:
                    raise ValueError("Please select exactly 2 hero cards")
                
                if len(self.selected_board_cards) != 3:
                    raise ValueError("Please select exactly 3 board cards")
                
                # Check if we have participating players
                participating = self.get_participating_players()
                if len(participating) < 2:
                    raise ValueError("Please select actions for at least 2 players in preflop")
                
                # Build scenario
                scenario = self.build_scenario()
                
                # Generate prompt with TOOL_TAGS (requires existing runner)
                if not hasattr(self, 'runner') or not self.runner:
                    raise RuntimeError("SystemC runner not initialized! Load model first.")
                
                prompt = self.runner.format_prompt(scenario)
                
                # Update prompt display on main thread
                def update_prompt():
                    self.prompt_display.delete(1.0, tk.END)
                    self.prompt_display.insert(tk.END, prompt)
                    
                    # Enable analyze button only if model is loaded
                    if self.model_loaded and hasattr(self, 'analyze_btn'):
                        self.analyze_btn.config(state=tk.NORMAL)
                    if self.model_loaded and hasattr(self, 'analyze_btn_results'):
                        self.analyze_btn_results.config(state=tk.NORMAL)
                    
                    if hasattr(self, 'status_var'):
                        if self.model_loaded:
                            self.status_var.set("Prompt generated successfully! Ready to analyze.")
                        else:
                            self.status_var.set("Prompt generated! Load model to enable analysis.")
                    
                    # Switch to results tab to show the prompt
                    try:
                        notebook = self.root.winfo_children()[0].winfo_children()[1]  # Get notebook
                        notebook.select(1)  # Select results tab
                    except:
                        pass  # Ignore if notebook structure changed
                
                self.root.after(0, update_prompt)
                
            except Exception as e:
                # Update GUI on main thread
                error_msg = str(e)  # Capture error message
                def update_error():
                    messagebox.showerror("Error", error_msg)
                    if hasattr(self, 'status_var'):
                        self.status_var.set("Failed to generate prompt")
                
                self.root.after(0, update_error)
            finally:
                # Re-enable button on main thread
                self.root.after(0, lambda: self.generate_prompt_btn.config(state=tk.NORMAL))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def build_scenario(self) -> Dict[str, str]:
        """Build scenario dictionary from GUI inputs."""
        # Convert hero cards
        hero_cards = []
        for card in self.selected_hero_cards:
            rank = card[0]
            suit = self.suit_symbols[card[1]]
            hero_cards.append(f"{rank}{suit}")
        
        # Convert board cards
        board_cards = []
        for card in self.selected_board_cards:
            rank = card[0]
            suit = self.suit_symbols[card[1]]
            board_cards.append(f"{rank}{suit}")
        
        # Build preflop history from actions
        preflop_history = []
        positions = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
        
        # Use the new preflop_actions array
        if hasattr(self, 'preflop_actions') and any(self.preflop_actions):
            for i, (pos, action) in enumerate(zip(positions, self.preflop_actions)):
                if action:  # If position has acted
                    if "Raise" in action:
                        size = action.split()[-1] if len(action.split()) > 1 else "2.5"
                        preflop_history.append(f"{pos} raises {size}bb")
                    elif action == "Call":
                        preflop_history.append(f"{pos} calls")
                    elif action == "Fold":
                        preflop_history.append(f"{pos} folds")
                    elif action == "All-in":
                        preflop_history.append(f"{pos} all-in")
                    else:
                        preflop_history.append(f"{pos} {action.lower()}")
            
            # Add response column actions to preflop history
            for response in self.response_columns:
                if response['action']:
                    pos_name = self.positions[response['position']]
                    action = response['action']
                    if "Raise" in action:
                        size = action.split()[-1] if len(action.split()) > 1 else "22"
                        preflop_history.append(f"{pos_name} raises {size}bb")
                    elif action == "Call":
                        preflop_history.append(f"{pos_name} calls")
                    elif action == "Fold":
                        preflop_history.append(f"{pos_name} folds")
                    elif action == "All-in":
                        preflop_history.append(f"{pos_name} all-in")
                    else:
                        preflop_history.append(f"{pos_name} {action.lower()}")
        else:
            # Default preflop scenario
            preflop_history.append("UTG folds, UTG+1 folds, LJ folds, HJ folds, CO folds, BTN raises 2.5bb, SB folds, BB calls")
        
        # Build postflop actions string
        postflop_actions_str = ""
        if hasattr(self, 'postflop_actions') and any(self.postflop_actions):
            postflop_sequence = []
            participating = self.get_participating_players()
            
            for i, action in enumerate(self.postflop_actions):
                if action and i < len(participating):
                    # Get actual position name instead of OOP/IP
                    pos_name = participating[i][0]  # participating is list of (pos_name, pos_index)
                    
                    # Format action with proper verb form
                    if action.lower() == "check":
                        action_str = f"{pos_name} checks"
                    elif "bet" in action.lower():
                        # Format betting actions properly
                        action_str = f"{pos_name} bets {action.split()[-1] if len(action.split()) > 1 else ''}"
                    elif action.lower() == "call":
                        action_str = f"{pos_name} calls"
                    elif action.lower() == "fold":
                        action_str = f"{pos_name} folds"
                    elif "all-in" in action.lower():
                        action_str = f"{pos_name} all-in"
                    else:
                        action_str = f"{pos_name} {action.lower()}"
                    
                    postflop_sequence.append(action_str)
            
            # Add postflop response actions
            for response in self.postflop_response_columns:
                if response['action'] and response['player'] < len(participating):
                    # Get actual position name instead of OOP/IP
                    pos_name = participating[response['player']][0]
                    action = response['action']
                    
                    # Format response action with proper verb form
                    if action.lower() == "check":
                        action_str = f"{pos_name} checks"
                    elif "bet" in action.lower():
                        action_str = f"{pos_name} bets {action.split()[-1] if len(action.split()) > 1 else ''}"
                    elif action.lower() == "call":
                        action_str = f"{pos_name} calls"
                    elif action.lower() == "fold":
                        action_str = f"{pos_name} folds"
                    elif "all-in" in action.lower():
                        action_str = f"{pos_name} all-in"
                    else:
                        action_str = f"{pos_name} {action.lower()}"
                    
                    postflop_sequence.append(action_str)
            
            postflop_actions_str = ', '.join(postflop_sequence)
        
        # Generate legal actions based on current state
        legal_actions = self.generate_legal_actions()
        
        scenario = {
            'hero_pos': self.hero_pos_var.get(),
            'hero_hand': ' '.join(hero_cards),
            'villain_profile': self.villain_profile_var.get(),
            'preflop_history': ', '.join(preflop_history),
            'flop': ' '.join(board_cards),
            'pot_size': str(self.calculated_pot),  # Use auto-calculated value
            'stacks': str(self.calculated_stacks),  # Use auto-calculated value
            'actions': postflop_actions_str,  # Use postflop actions instead of manual entry
            'legal_actions': legal_actions
        }
        
        return scenario
    
    def generate_legal_actions(self):
        """Generate legal actions based on current postflop state."""
        if not hasattr(self, 'postflop_actions') or not any(self.postflop_actions):
            # Default legal actions for first to act (OOP)
            pot_size = self.calculated_pot if self.calculated_pot > 0 else 6.5
            bet_33 = pot_size * 0.33
            bet_50 = pot_size * 0.5
            bet_100 = pot_size
            
            return f"[check,bet 33% ({bet_33:.1f}bb),bet 50% ({bet_50:.1f}bb),bet 100% ({bet_100:.1f}bb),allin]"
        
        # Determine current state and generate appropriate actions
        participating = self.get_participating_players()
        if len(participating) < 2:
            return "[check,bet 33%,bet 50%,bet 100%,allin]"
        
        oop_action = self.postflop_actions[0] if len(self.postflop_actions) > 0 else ""
        ip_action = self.postflop_actions[1] if len(self.postflop_actions) > 1 else ""
        
        pot_size = self.calculated_pot if self.calculated_pot > 0 else 6.5
        
        if not oop_action:
            # OOP to act first
            bet_33 = pot_size * 0.33
            bet_50 = pot_size * 0.5
            bet_100 = pot_size
            return f"[check,bet 33% ({bet_33:.1f}bb),bet 50% ({bet_50:.1f}bb),bet 100% ({bet_100:.1f}bb),allin]"
        elif oop_action == "Check" and not ip_action:
            # IP to act after check
            bet_33 = pot_size * 0.33
            bet_50 = pot_size * 0.5
            bet_100 = pot_size
            return f"[check,bet 33% ({bet_33:.1f}bb),bet 50% ({bet_50:.1f}bb),bet 100% ({bet_100:.1f}bb),allin]"
        elif "Bet" in oop_action and not ip_action:
            # IP to respond to bet
            return "[fold,call,raise 75%,allin]"
        else:
            # Check for response columns
            if self.postflop_response_columns:
                return "[fold,call,raise 75%,allin]"
            else:
                return "[check,bet 33%,bet 50%,bet 100%,allin]"
    
    def clear_all(self):
        """Clear all inputs and reset to default state."""
        # Clear cards
        self.selected_hero_cards.clear()
        self.selected_board_cards.clear()
        self.update_card_displays()
        
        # Reset position and profile
        if hasattr(self, 'hero_pos_var'):
            self.hero_pos_var.set("BTN")
        if hasattr(self, 'villain_profile_var'):
            self.villain_profile_var.set("balanced")
        
        # Reset preflop actions
        if hasattr(self, 'preflop_actions'):
            self.preflop_actions = [""] * 8  # Clear all actions
            self.preflop_aggressor = None
            self.preflop_aggressor_group = None
            self.action_to_act = 0
            self.betting_round = 1  # Reset betting round
            self.clear_response_columns()  # Clear any response columns
            self.update_preflop_display()  # Refresh buttons and title
        
        # Reset postflop actions
        if hasattr(self, 'postflop_actions'):
            self.postflop_actions = ["", ""]
            self.clear_postflop_response_columns()
            if hasattr(self, 'postflop_frame'):
                self.update_postflop_display()
        
        # Reset calculated values
        self.calculated_pot = 0.0
        self.calculated_stacks = 100.0
        if hasattr(self, 'pot_display'):
            self.pot_display.config(text="0.0")
        if hasattr(self, 'stacks_display'):
            self.stacks_display.config(text="100.0")
        
        # Clear displays
        if hasattr(self, 'prompt_display'):
            self.prompt_display.delete(1.0, tk.END)
        if hasattr(self, 'response_display'):
            self.response_display.delete(1.0, tk.END)
        
        # Reset button states based on model loading status
        if hasattr(self, 'analyze_btn_results'):
            # Only enable if model is loaded
            state = tk.NORMAL if self.model_loaded else tk.DISABLED
            self.analyze_btn_results.config(state=tk.DISABLED)  # Disabled until prompt is generated
        
        if hasattr(self, 'status_var'):
            self.status_var.set("All inputs cleared")
    
    def run(self):
        """Start the GUI."""
        # Check dependencies
        if not DEPENDENCIES_AVAILABLE:
            messagebox.showerror("Missing Dependencies", 
                               "Required ML libraries not found. Please install:\n"
                               "pip install torch transformers peft accelerate")
            return
        
        # Bring window to front on macOS
        try:
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
        except:
            pass
        
        # Bind mouse wheel to canvas
        def _on_mousewheel(event):
            # Only scroll if the widget is a Canvas
            widget = event.widget
            if hasattr(widget, 'yview_scroll'):
                try:
                    widget.yview_scroll(int(-1*(event.delta/120)), "units")
                except:
                    pass  # Ignore scroll errors
        
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Auto-load model after GUI is set up
        self.root.after(100, self.load_model_thread)
        
        # Start GUI
        self.root.mainloop()

    def add_response_column(self, position_index, response_type):
        """Add a response column for 3bet/4bet situations."""
        position_name = self.positions[position_index]
        
        # Check if this response column already exists
        for response in self.response_columns:
            if response['position'] == position_index and response['type'] == response_type:
                return  # Already exists
        
        # Create new response column frame in the action_frame
        response_frame = ttk.Frame(self.action_frame)
        response_frame.pack(side=tk.LEFT, padx=5)
        
        # Position label - FIXED SIZING TO MATCH DEFAULT BOXES
        pos_label = tk.Label(response_frame, text=position_name, 
                           font=('Arial', 9, 'bold'),  # Match main position labels
                           width=10, height=2,         # Match main position labels  
                           bg='lightblue', fg='black',
                           relief='solid', borderwidth=1)
        pos_label.pack(pady=1)
        
        # Button frame for this response column
        button_frame = ttk.Frame(response_frame)
        button_frame.pack()
        
        # Response actions based on type
        if response_type == "3bet_response":
            response_actions = ["Fold", "Call", "Raise 22"]  # 4bet to 22bb
        else:
            response_actions = ["Fold", "Call", "All-in"]
        
        # Create buttons for response actions - FIXED SIZING
        response_buttons = []
        for action in response_actions:
            btn = tk.Button(button_frame, 
                          text=action,
                          font=('Arial', 8, 'bold'),   # Match main buttons
                          width=10, height=1,          # Match main buttons
                          bg='white', fg='black',
                          relief='solid', borderwidth=1,
                          command=lambda pos=position_index, act=action, resp_type=response_type: 
                                 self.select_response_action(pos, act, resp_type))
            btn.pack(pady=1)
            response_buttons.append(btn)
        
        # Store response column info
        response_info = {
            'position': position_index,
            'type': response_type,
            'frame': response_frame,
            'button_frame': button_frame,
            'buttons': response_buttons,
            'action': ""
        }
        self.response_columns.append(response_info)
    
    def select_response_action(self, position_index, action, response_type):
        """Handle response action selection (3bet/4bet response)."""
        # Find the correct response column
        for response in self.response_columns:
            if response['position'] == position_index and response['type'] == response_type:
                response['action'] = action
                
                # Update button styling - SAME AS PREFLOP
                for btn in response['buttons']:
                    btn_text = btn.cget('text')
                    if btn_text == action:
                        btn.config(bg='white', fg='black', state=tk.NORMAL)
                    else:
                        btn.config(bg='#f0f0f0', fg='#888888', state=tk.DISABLED)
                
                # If this is a 4bet (Raise 22), add another response column for the 3bettor
                if "Raise" in action and response_type == "3bet_response":
                    # Find who the 3bettor was
                    three_bettor = None
                    for i, act in enumerate(self.preflop_actions):
                        if act and "Raise" in act and i != self.preflop_aggressor:
                            three_bettor = i
                            break
                    
                    if three_bettor is not None:
                        self.add_response_column(three_bettor, "4bet_response")
                
                break
        
        # Update display
        self.update_preflop_display()
    
    def clear_response_columns(self):
        """Clear all response columns."""
        for response in self.response_columns:
            response['frame'].destroy()
        self.response_columns.clear()

def main():
    """Main function."""
    # Check if we're in the right conda environment
    import sys
    import os
    
    # Check if we're in the poker-llm environment
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env != 'poker-llm':
        print("\n" + "="*60)
        print("⚠️  ENVIRONMENT WARNING")
        print("="*60)
        print("For full functionality, activate the poker-llm conda environment:")
        print()
        print("📋 Run these commands:")
        print("   conda activate poker-llm")
        print("   python poker_gui.py")
        print()
        print("💡 Or create a simple launcher script:")
        print("   echo '#!/bin/bash' > run_gui.sh")
        print("   echo 'conda activate poker-llm && python poker_gui.py' >> run_gui.sh")
        print("   chmod +x run_gui.sh")
        print("   ./run_gui.sh")
        print()
        print(f"Current environment: {current_env or 'base'}")
        print("Recommended environment: poker-llm")
        print()
        print("🚀 Starting GUI anyway (model loading may fail)...")
        print("="*60)
        # Continue instead of exiting
    
    app = PokerSolverGUI()
    app.run()

if __name__ == "__main__":
    main()