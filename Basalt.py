import numpy as np
import os
import readgssi.readgssi as dzt
import re

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from scipy import signal
from math import sqrt, floor
from scipy.ndimage import uniform_filter1d

import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QListWidget, QPushButton, QComboBox, QLineEdit, QTabWidget, QCheckBox, QSlider, QListWidgetItem
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtGui import QDoubleValidator, QValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from typing import Union

cste_global = {
    "c_lum": 299792458, # Vitesse de la lumière dans le vide en m/s
    }
class Radar(Enum):
    MALA = 0
    GSSI_XT = 1
    GSSI_FLEX = 2

@dataclass
class TraitementValues:
    t0_lin :int = 0
    t0_exp: int = 0
    a_lin : float = 0.0
    a_exp : float = 0.0
    g : float = 1.0
    t_max_exp: int = 0

    is_dewow_active: bool = False

    is_sub_mean: bool = False
    sub_mean_mode: str = 'Globale' # 'Globale' ou 'Mobile'
    sub_mean_window: int = 21

    is_filtre_freq : bool = False
    antenna_freq : float = 0 
    sampling_freq : float = 0

    T0 : int = 0 #borne basse Y
    T_lim : int = None # borne haute Y
    X0 : int = 0 #borne basse X
    X_lim : int = None # borne haute X

    X_dist : int = None #Forcer la taille du fichier 

    X_ticks : int = 20 #Nombre de repère verticaux
    Y_ticks : int = 20 #nombre de repère horizontaux
    show_x_ticks: bool = False
    show_y_ticks: bool = False

    epsilon :float = 8.0
    contraste :float= 1.0

    unit_x: str = "Distance (m)"
    unit_y: str = "Profondeur (m)"

 
class Const():
    def __init__(self):
        self.ext_list = [".rd7", ".rd3", ".DZT",".dzt"]
        self.freq_state = ["Filtrage désactivé", "Haute Fréquence", "Basse Fréquence"]
        
        self.Xunit = ["Distance", "Temps", "Traces"]
        self.Yunit = ["Profondeur", "Temps", "Samples"]
        self.Xlabel = ["Distance (m)", "Temps (s)", "Traces"]
        self.Ylabel = ["Profondeur (m)", "Temps (ns)", "Samples"]
        self.xLabel = ["m", "s", "mesures"]
        self.yLabel = ["m", "ns", "samples"]


    def getRadarByExtension(self, ext:str):
        match ext:
            case ".rd7"  | ".rd3":
                return Radar.MALA
            case ".DZT":
                return Radar.GSSI_XT
            case ".dzt":
                return Radar.GSSI_FLEX
    def getFiltreFreq(self, freq:str):
        match freq:
            case "Filtrage désactivé":
                return None
            case "Haute Fréquence":
                return "_1"
            case "Basse Fréquence":
                return '_2'
            case _:
                return None
            
    def getFiltreExtension(self,ext:str):
        return "*" + ext

class AcceptEmptyDoubleValidator(QDoubleValidator):
    def validate(self, input_str: str, pos: int) -> tuple['QValidator.State', str, int]:
        # Si la chaîne de caractères est vide...
        if not input_str:
            # ... on dit à Qt que c'est un état parfaitement ACCEPTABLE.
            return (QValidator.State.Acceptable, input_str, pos)
        
        # Pour tous les autres cas (texte non vide), on utilise le comportement
        # normal du QDoubleValidator parent.
        return super().validate(input_str, pos)
    
class MainWindow():
    def __init__(self, softwarename:str):
        self.constante = Const()

        # Création de notre fenêtre principale
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle(softwarename)
        
        self.basalt :Basalt = Basalt()
        
        # UI 
        self.window.setGeometry(100, 100, 1600, 900)
        self.window.central_widget = QWidget()
        self.window.setCentralWidget(self.window.central_widget)
        self.main_layout = QHBoxLayout(self.window.central_widget)

                # --- CRÉATION DU PANNEAU DE CONTRÔLE DE GAUCHE ---
        # Ce widget contiendra tous les boutons, sliders, etc.
        self.control_panel_widget = QWidget()
        # On donne à ce widget son propre layout vertical
        self.control_layout = QVBoxLayout(self.control_panel_widget)
        # On peut fixer la largeur du panneau de contrôle pour un meilleur design
        self.control_panel_widget.setFixedWidth(350)
        self.init_ui()

        self.menu()

        # --- CRÉATION DU GRAPHIQUE (PANNEAU DE DROITE) ---
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.radargramme = Graphique(self.ax, self.fig)

                # --- AJOUT DES DEUX PANNEAUX AU LAYOUT PRINCIPAL ---
        # 1. On ajoute le panneau de contrôle (qui contient tous les petits widgets)
        self.main_layout.addWidget(self.control_panel_widget)

        # 2. On ajoute le canvas du graphique
        self.main_layout.addWidget(self.canvas)
        
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        # Donne plus d'espace au graphique quand la fenêtre est redimensionnée.
        #self.main_layout.setStretchFactor(0, 1) # Index 0 : control_panel_widget
        #self.main_layout.setStretchFactor(1, 3) # Index 1 : canvas

    def show(self):
        self.window.show()
        sys.exit(self.app.exec())

    def menu(self):
        # Création de la barre de menu
        menu_bar = self.window.menuBar()
        
        # Création des différents Menus
        file_menu = menu_bar.addMenu("Fichier")

        # Création des actions pour le menu "Fichier"
        open_folder_action = QAction("Ouvrir un dossier", self.window)
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)

        save_img_action = QAction("Sauvegarder l'image", self.window)
        save_img_action.triggered.connect(self.exportPNG)
        file_menu.addAction(save_img_action)

        save_imgs_action = QAction("Sauvegarder les images", self.window)
        save_imgs_action.triggered.connect(self.exportPNG_all)
        file_menu.addAction(save_imgs_action)
   
    def init_ui(self):
        #---Liste des fichier
        self.listFile_widget = QListWidget()
        self.control_layout.addWidget(self.listFile_widget)
        self.listFile_widget.itemClicked.connect(self.on_file_clicked)
        #        ---

        ## Ajout des QComboBox de filtre fichier et antenne

# Crée un layout horizontal pour la première paire (Extension)
        extension_layout = QHBoxLayout()
        label_extension = QLabel("Extension de fichier:")
        self.combo_box_extension = QComboBox()
        self.combo_box_extension.addItems(self.constante.ext_list)
        # Ajoute le label et la combobox à LEUR propre layout horizontal
        extension_layout.addWidget(label_extension)
        extension_layout.addWidget(self.combo_box_extension)
        # Ajoute ce premier layout de paire au panneau de contrôle vertical
        self.control_layout.addLayout(extension_layout)

        # Crée un layout horizontal pour la deuxième paire (Fréquence)
        frequence_layout = QHBoxLayout()
        label_frequence = QLabel("Filtre fréquence :")
        self.combo_box_frequence = QComboBox()
        self.combo_box_frequence.addItems(self.constante.freq_state)
        # Ajoute le label et la combobox à LEUR propre layout horizontal
        frequence_layout.addWidget(label_frequence)
        frequence_layout.addWidget(self.combo_box_frequence)
        # Ajoute ce deuxième layout de paire au panneau de contrôle vertical
        self.control_layout.addLayout(frequence_layout)


        # On connecte les signaux après avoir créé les objets
        self.combo_box_extension.currentIndexChanged.connect(self.populate_listFile_widget)
        self.combo_box_frequence.currentIndexChanged.connect(self.populate_listFile_widget)
        # ----- 

        ## Réglage du Contraste

        # Layout horizontal pour le slider de contraste
        self.contrast_layout = QHBoxLayout()
        self.control_layout.addLayout(self.contrast_layout)

        # Label pour le contraste
        self.label_contrast = QLabel("Contraste:")
        self.contrast_layout.addWidget(self.label_contrast)

        # QSlider pour le contraste
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setMinimum(1)    # Correspond à 0.01 (0.01 * 100)
        self.slider_contrast.setMaximum(100)  # Correspond à 1.00 (1.00 * 100)
        self.slider_contrast.setValue(100)     # Valeur initiale, correspond à 0.50
        self.slider_contrast.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_contrast.setTickInterval(10) # Affiche des marques tous les 0.1 (10 unités)
        self.contrast_layout.addWidget(self.slider_contrast)

        # Label pour afficher la valeur actuelle du contraste
        self.current_contrast_label = QLabel("1.0") # Texte initial basé sur slider.setValue(50)
        self.contrast_layout.addWidget(self.current_contrast_label)

        # Connecter le signal valueChanged du slider
        self.slider_contrast.valueChanged.connect(self.on_contrast_slider_changed)

        # ---
        
        ## Saisie des Paramètres 

        # Layout horizontal pour les champs de texte
        self.params_layout = QHBoxLayout()
        self.control_layout.addLayout(self.params_layout)

        # Création du validateur pour les nombres flottants (double)
        # Il accepte des nombres entre -1000 et 1000 (ajuste les limites si besoin)
        # et avec 2 chiffres après la virgule.
        self.float_validator = QDoubleValidator(0, 100000.0, 2)
        self.float_validator = AcceptEmptyDoubleValidator(0, 100000.0, 2)
        # Permet la notation standard (ex: 12.34), pas la notation scientifique
        self.float_validator.setNotation(QDoubleValidator.Notation.StandardNotation) 

        # Champ de texte pour la Distance
        self.label_distance = QLabel("Distance (m):")
        self.params_layout.addWidget(self.label_distance)
        self.line_edit_distance = QLineEdit()
        self.line_edit_distance.setPlaceholderText("Forcer la distance (optionnel)")
        self.line_edit_distance.setValidator(self.float_validator) # Applique le validateur
        self.params_layout.addWidget(self.line_edit_distance)
        self.line_edit_distance.editingFinished.connect(self.on_distance_edited)

        # Champ de texte pour Epsilon
        self.label_epsilon = QLabel("Epsilon:")
        self.params_layout.addWidget(self.label_epsilon)
        self.line_edit_epsilon = QLineEdit()
        self.line_edit_epsilon.setPlaceholderText("Entrez epsilon")
        self.line_edit_epsilon.setText("8.0") # Valeur par défaut
        self.line_edit_epsilon.setValidator(self.float_validator) # Applique le validateur
        self.params_layout.addWidget(self.line_edit_epsilon)
        # Connecter le signal editingFinished pour Epsilon
        self.line_edit_epsilon.editingFinished.connect(self.on_epsilon_edited)

    # ---- 
         ## Unités des Axes ----- Aucune fonction d'évènement
        # Layout horizontal pour les ComboBox des unités
        self.units_layout = QHBoxLayout()
        self.control_layout.addLayout(self.units_layout)

        # ComboBox pour l'unité de l'abscisse (X)
        self.label_unit_x = QLabel("Unité X:")
        self.units_layout.addWidget(self.label_unit_x)
        self.combo_box_unit_x = QComboBox()
        self.combo_box_unit_x.addItems(self.constante.Xlabel)
        self.units_layout.addWidget(self.combo_box_unit_x)
        self.combo_box_unit_x.currentTextChanged.connect(self.on_unit_x_changed)

        # ComboBox pour l'unité de l'ordonnée (Y)
        self.label_unit_y = QLabel("Unité Y:")
        self.units_layout.addWidget(self.label_unit_y)
        self.combo_box_unit_y = QComboBox()
        self.combo_box_unit_y.addItems(self.constante.Ylabel)
        self.units_layout.addWidget(self.combo_box_unit_y)
        self.combo_box_unit_y.currentTextChanged.connect(self.on_unit_y_changed)

        # --- 

        ## Limites des Axes (X0, X_lim, Y0, Y_lim)      
        # Layout horizontal pour les limites de l'axe X
        self.xlim_layout = QHBoxLayout()
        self.control_layout.addLayout(self.xlim_layout)

        self.label_x0 = QLabel("X0:")
        self.xlim_layout.addWidget(self.label_x0)
        self.line_edit_x0 = QLineEdit()
        self.line_edit_x0.setPlaceholderText("Début X")
        self.line_edit_x0.setText("0.0")
        self.line_edit_x0.setValidator(self.float_validator)
        self.xlim_layout.addWidget(self.line_edit_x0)
        self.line_edit_x0.editingFinished.connect(self.on_x0_edited)

        self.label_xlim = QLabel("X_lim:")
        self.xlim_layout.addWidget(self.label_xlim)
        self.line_edit_xlim = QLineEdit()
        self.line_edit_xlim.setPlaceholderText("Fin X")
        self.line_edit_xlim.setValidator(self.float_validator)
        self.xlim_layout.addWidget(self.line_edit_xlim)
        self.line_edit_xlim.editingFinished.connect(self.on_xlim_edited)

        # Layout horizontal pour les limites de l'axe Y
        self.ylim_layout = QHBoxLayout()
        self.control_layout.addLayout(self.ylim_layout)

        self.label_y0 = QLabel("Y0:")
        self.ylim_layout.addWidget(self.label_y0)
        self.line_edit_y0 = QLineEdit()
        self.line_edit_y0.setPlaceholderText("Début Y")
        self.line_edit_y0.setText("0.0")
        self.line_edit_y0.setValidator(self.float_validator)
        self.ylim_layout.addWidget(self.line_edit_y0)
        self.line_edit_y0.editingFinished.connect(self.on_y0_edited)

        self.label_ylim = QLabel("Y_lim:")
        self.ylim_layout.addWidget(self.label_ylim)
        self.line_edit_ylim = QLineEdit()
        self.line_edit_ylim.setPlaceholderText("Fin Y")
        self.line_edit_ylim.setValidator(self.float_validator)
        self.ylim_layout.addWidget(self.line_edit_ylim)
        self.line_edit_ylim.editingFinished.connect(self.on_ylim_edited)
        self.line_edit_ylim.returnPressed.connect(self.on_ylim_edited)

    # ---
        ## Section Multi-pages (QTabWidget)
        self.tab_widget = QTabWidget()
        self.control_layout.addWidget(self.tab_widget) # Ajoute le QTabWidget au layout principal

        # Création et ajout de la première page (Paramètres de Gain)
        self.gain_page = self.create_gain_page()
        self.tab_widget.addTab(self.gain_page, "Paramètres de Gain")

        # Création et ajout de la  page (Filtres)
        self.filter_page = self.create_filter_page()
        self.tab_widget.addTab(self.filter_page, "Filtres")

        # Création et ajout de la page (Options)
        self.options_page = self.create_options_page()
        self.tab_widget.addTab(self.options_page, "Options")
        # -------------------------

        # Création et ajout de la page (Infos)
        self.info_page = self.create_info_page()
        self.tab_widget.addTab(self.info_page, "Infos")

        #---- Pointeur
        # Création d'un cadre pour séparer visuellement
        coord_frame = QFrame()
        coord_frame.setFrameShape(QFrame.Shape.StyledPanel)
        coord_frame_layout = QVBoxLayout(coord_frame)
        
        # Label pour les coordonnées
        self.coord_label = QLabel("X: -- | Y: --")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Centrer le texte
        coord_frame_layout.addWidget(self.coord_label)

        # On ajoute ce cadre au layout principal du panneau de contrôle
        self.control_layout.addWidget(coord_frame)
        #----------------------------

        # Ajoutez d'autres pages ici si nécessaire (ex: Filtrage, Affichage, etc.)
        # self.other_page = QWidget()
        # self.tab_widget.addTab(self.other_page, "Autre Page")

        # Ajoutez un espaceur pour pousser les éléments vers le haut si nécessaire
        self.control_layout.addStretch()
    

    def redraw_plot(self):
        """
        Fonction LÉGÈRE : redessine le graphique avec les données déjà traitées.
        Idéal pour les changements qui n'affectent que l'affichage (contraste, grille...).
        """
        if self.basalt.traitement is None or self.basalt.traitement.data is None:
            return

        print("Rafraîchissement du graphique...")
        
        # --- CORRECTION CLÉ ---
        # On met à jour l'attribut contraste de l'objet Graphique AVANT de tracer
        self.radargramme.contraste = self.basalt.traitementValues.contraste
        # ----------------------

        # On récupère les autres paramètres d'affichage
        plot_extent, xlabel, ylabel = self.basalt.get_plot_axes_parameters()
        if plot_extent is None: return

        # On appelle la fonction de tracé
        self.radargramme.plot(data=self.basalt.traitement.data,
                            title=self.basalt.selectedFile.stem,
                            x_ticks=self.basalt.traitementValues.X_ticks,
                            y_ticks=self.basalt.traitementValues.Y_ticks,
                            extent=plot_extent,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            show_x_ticks=self.basalt.traitementValues.show_x_ticks,
                            show_y_ticks=self.basalt.traitementValues.show_y_ticks)
                            
        self.canvas.draw()

    def update_display(self):
        """
        Fonction COÛTEUSE : relance tous les traitements PUIS appelle le redessin.
        À utiliser pour les changements de gain, de filtre, de limites, etc.
        """
        if self.basalt.data is None:
            return

        print("Recalcul des données de traitement...")
        # 1. Relancer le traitement complet
        self.basalt.traitementScan()
        # 2. Appeler la fonction de redessin pour afficher le résultat
        self.redraw_plot()
        
        
    def create_gain_page(self):
        """Crée et retourne la page des paramètres de gain."""
        gain_widget = QWidget()
        gain_layout = QVBoxLayout(gain_widget) # Un layout vertical pour cette page

        # Titre pour la section Gain Constant
        label_constant_gain = QLabel("--- Gain constant ---")
        label_constant_gain.setStyleSheet("font-weight: bold; margin-top: 10px;")
        gain_layout.addWidget(label_constant_gain)
        
        # Paramètre g
        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel("g:"))
        self.line_edit_g = QLineEdit()
        self.line_edit_g.setPlaceholderText("Coefficient g")
        self.line_edit_g.setText("1") # Valeur par défaut
        self.line_edit_g.setValidator(self.float_validator)
        g_layout.addWidget(self.line_edit_g)
        self.line_edit_g.editingFinished.connect(self.on_g_edited)
        gain_layout.addLayout(g_layout)

        # Titre pour la section Gain Linéaire
        label_linear_gain = QLabel("--- Gain Linéaire ---")
        label_linear_gain.setStyleSheet("font-weight: bold; margin-top: 10px;")
        gain_layout.addWidget(label_linear_gain)

  
        # Paramètre a_lin
        a_lin_layout = QHBoxLayout()
        a_lin_layout.addWidget(QLabel("a_lin:"))
        self.line_edit_a_lin = QLineEdit()
        self.line_edit_a_lin.setPlaceholderText("Amplitude linéaire")
        self.line_edit_a_lin.setText("0.0") # Valeur par défaut
        self.line_edit_a_lin.setValidator(self.float_validator)
        a_lin_layout.addWidget(self.line_edit_a_lin)
        self.line_edit_a_lin.editingFinished.connect(self.on_a_lin_edited)
        gain_layout.addLayout(a_lin_layout)

        # Paramètre t0_lin
        t0_lin_layout = QHBoxLayout()
        t0_lin_layout.addWidget(QLabel("t0_lin:"))
        self.line_edit_t0_lin = QLineEdit()
        self.line_edit_t0_lin.setPlaceholderText("Temps initial linéaire")
        self.line_edit_t0_lin.setText("0.0") # Valeur par défaut
        self.line_edit_t0_lin.setValidator(self.float_validator)
        t0_lin_layout.addWidget(self.line_edit_t0_lin)
        self.line_edit_t0_lin.editingFinished.connect(self.on_t0_lin_edited)
        gain_layout.addLayout(t0_lin_layout)

        # Espaceur pour séparer les sections de gain
        gain_layout.addSpacing(20)

        # Titre pour la section Gain Exponentiel
        label_exp_gain = QLabel("--- Gain Exponentiel ---")
        label_exp_gain.setStyleSheet("font-weight: bold;")
        gain_layout.addWidget(label_exp_gain)

        # Paramètre a_exp
        a_exp_layout = QHBoxLayout()
        a_exp_layout.addWidget(QLabel("a_exp:"))
        self.line_edit_a_exp = QLineEdit()
        self.line_edit_a_exp.setPlaceholderText("Amplitude exponentielle")
        self.line_edit_a_exp.setText("1.0") # Valeur par défaut
        self.line_edit_a_exp.setValidator(self.float_validator)
        a_exp_layout.addWidget(self.line_edit_a_exp)
        self.line_edit_a_exp.editingFinished.connect(self.on_a_exp_edited)
        gain_layout.addLayout(a_exp_layout)

        # Paramètre T0_exp
        T0_exp_layout = QHBoxLayout()
        T0_exp_layout.addWidget(QLabel("T0_exp:"))
        self.line_edit_T0_exp = QLineEdit()
        self.line_edit_T0_exp.setPlaceholderText("Temps initial exponentiel")
        self.line_edit_T0_exp.setText("0.0") # Valeur par défaut
        self.line_edit_T0_exp.setValidator(self.float_validator)
        T0_exp_layout.addWidget(self.line_edit_T0_exp)
        self.line_edit_T0_exp.editingFinished.connect(self.on_T0_exp_edited)
        gain_layout.addLayout(T0_exp_layout)

        #Paramètre T_max_exp
        t_max_exp_layout = QHBoxLayout()
        t_max_exp_layout.addWidget(QLabel("t_max_exp:"))
        self.line_edit_t_max_exp = QLineEdit()
        self.line_edit_t_max_exp.setPlaceholderText("Fin du gain expo (sample), 0=fin trace")
        self.line_edit_t_max_exp.setText("0") # Par défaut, 0 pour aller jusqu'au bout
        self.line_edit_t_max_exp.setValidator(self.float_validator) # Peut-être utiliser un QIntValidator ici
        t_max_exp_layout.addWidget(self.line_edit_t_max_exp)
        self.line_edit_t_max_exp.editingFinished.connect(self.on_t_max_exp_edited)
        gain_layout.addLayout(t_max_exp_layout)

        gain_layout.addStretch() # Pousse tous les éléments vers le haut

        return gain_widget
    
    def create_filter_page(self):
        """Crée et retourne la page des paramètres de filtre."""
        filter_widget = QWidget()
        filter_layout = QVBoxLayout(filter_widget)

        # Section Dewow
        dewow_layout = QHBoxLayout()
        self.checkbox_dewow = QCheckBox("Activer dewow")
        self.checkbox_dewow.setChecked(False) 
        self.checkbox_dewow.stateChanged.connect(self.on_dewow_toggled)
        dewow_layout.addWidget(self.checkbox_dewow)
        dewow_layout.addStretch() # Pousse le checkbox à gauche
        filter_layout.addLayout(dewow_layout)

        # --- NOUVEAU BLOC À AJOUTER (dans create_filter_page) ---
        label_submean = QLabel("--- Retrait de la Trace Moyenne ---")
        label_submean.setStyleSheet("font-weight: bold; margin-top: 10px;")
        filter_layout.addWidget(label_submean)

        # 1. Checkbox pour activer/désactiver la fonction
        self.checkbox_sub_mean = QCheckBox("Activer le retrait de trace moyenne")
        self.checkbox_sub_mean.setChecked(False) # Désactivé par défaut
        filter_layout.addWidget(self.checkbox_sub_mean)

        # Layout pour les options
        sub_mean_options_layout = QHBoxLayout()

        # 2. ComboBox pour choisir le mode
        self.combo_sub_mean_mode = QComboBox()
        self.combo_sub_mean_mode.addItems(["Globale", "Mobile"])
        sub_mean_options_layout.addWidget(QLabel("Mode :"))
        sub_mean_options_layout.addWidget(self.combo_sub_mean_mode)

        # 3. LineEdit pour la taille de la fenêtre (pour le mode Mobile)
        self.line_edit_sub_mean_window = QLineEdit()
        self.line_edit_sub_mean_window.setPlaceholderText("Taille fenêtre")
        self.line_edit_sub_mean_window.setText("21") # Une valeur impaire est souvent utilisée
        self.line_edit_sub_mean_window.setValidator(QDoubleValidator(1, 1001, 0)) # Validateur d'entiers
        sub_mean_options_layout.addWidget(QLabel("Fenêtre :"))
        sub_mean_options_layout.addWidget(self.line_edit_sub_mean_window)

        filter_layout.addLayout(sub_mean_options_layout)
        
        # On désactive les options au départ
        self.combo_sub_mean_mode.setEnabled(False)
        self.line_edit_sub_mean_window.setEnabled(False)

        # Connexion des signaux aux nouvelles fonctions de rappel
        self.checkbox_sub_mean.stateChanged.connect(self.on_sub_mean_toggled)
        self.combo_sub_mean_mode.currentIndexChanged.connect(self.on_sub_mean_mode_changed)
        self.line_edit_sub_mean_window.editingFinished.connect(self.on_sub_mean_window_edited)

        # Filtre Fréquentiel (Passe-Haut / Passe-Bas)
        label_freq_filter = QLabel("--- Filtre fréquentiel ---")
        label_freq_filter.setStyleSheet("font-weight: bold; margin-top: 10px;")
        filter_layout.addWidget(label_freq_filter)
        
        self.checkbox_filtre_freq = QCheckBox("Activer le filtre fréquentiel")
        self.checkbox_filtre_freq.setChecked(False) # Le filtre est désactivé par défaut
        filter_layout.addWidget(self.checkbox_filtre_freq)
        
        # On connecte le changement d'état à une nouvelle fonction
        self.checkbox_filtre_freq.stateChanged.connect(self.on_filtre_freq_changed)

        freq_filtre_layout = QHBoxLayout()
        freq_filtre_layout.addWidget(QLabel("Sampling freq:"))
        self.line_edit_freq_filtre = QLineEdit()
        self.line_edit_freq_filtre.setPlaceholderText("Fréquence d'acquisition")
        self.line_edit_freq_filtre.setValidator(self.float_validator)
        freq_filtre_layout.addWidget(self.line_edit_freq_filtre)
        self.line_edit_freq_filtre.editingFinished.connect(self.on_freq_filtre_edited)
        filter_layout.addLayout(freq_filtre_layout)

        antenna_freq_layout = QHBoxLayout()
        antenna_freq_layout.addWidget(QLabel("Antenna freq"))
        self.line_edit_antenna_freq = QLineEdit()
        self.line_edit_antenna_freq.setPlaceholderText("Fréquence de l'antenne")
        self.line_edit_antenna_freq.setValidator(self.float_validator)
        antenna_freq_layout.addWidget(self.line_edit_antenna_freq)
        self.line_edit_antenna_freq.editingFinished.connect(self.on_cuttoff_freq_edited)
        filter_layout.addLayout(antenna_freq_layout)

        filter_layout.addStretch() # Pousse les éléments vers le haut

        return filter_widget

    def create_info_page(self):
        pass

    def create_options_page(self):
        """Crée et retourne la page des options d'affichage."""
        options_widget = QWidget()
        options_layout = QVBoxLayout(options_widget) # Le layout vertical principal de l'onglet

        # --- Section pour les graduations de l'axe X ---
        label_xticks = QLabel("--- Graduations de l'axe X (Grille Verticale) ---")
        label_xticks.setStyleSheet("font-weight: bold; margin-top: 10px;")
        options_layout.addWidget(label_xticks)

        # 1. On met la checkbox directement dans le layout vertical
        self.checkbox_show_x_ticks = QCheckBox("Afficher la Grille Verticale")
        self.checkbox_show_x_ticks.setChecked(False)
        options_layout.addWidget(self.checkbox_show_x_ticks)

        # 2. On crée un layout horizontal juste pour le champ "Nombre"
        x_ticks_number_layout = QHBoxLayout()
        x_ticks_number_layout.addWidget(QLabel("Nombre de lignes :"))
        self.line_edit_x_ticks = QLineEdit()
        self.line_edit_x_ticks.setText(str(self.basalt.traitementValues.X_ticks))
        self.line_edit_x_ticks.setValidator(QDoubleValidator(1, 100, 0))
        x_ticks_number_layout.addWidget(self.line_edit_x_ticks)
        # On ajoute ce layout horizontal au layout vertical principal
        options_layout.addLayout(x_ticks_number_layout)


        # --- Section pour les graduations de l'axe Y ---
        label_yticks = QLabel("--- Graduations de l'axe Y (Grille Horizontale) ---")
        label_yticks.setStyleSheet("font-weight: bold; margin-top: 20px;")
        options_layout.addWidget(label_yticks)
        
        # 1. On met la checkbox directement dans le layout vertical
        self.checkbox_show_y_ticks = QCheckBox("Afficher la Grille Horizontale")
        self.checkbox_show_y_ticks.setChecked(False)
        options_layout.addWidget(self.checkbox_show_y_ticks)

        # 2. On crée un layout horizontal juste pour le champ "Nombre"
        y_ticks_number_layout = QHBoxLayout()
        y_ticks_number_layout.addWidget(QLabel("Nombre de lignes :"))
        self.line_edit_y_ticks = QLineEdit()
        self.line_edit_y_ticks.setText(str(self.basalt.traitementValues.Y_ticks))
        self.line_edit_y_ticks.setValidator(QDoubleValidator(1, 100, 0))
        y_ticks_number_layout.addWidget(self.line_edit_y_ticks)
        # On ajoute ce layout horizontal au layout vertical principal
        options_layout.addLayout(y_ticks_number_layout)


        # La connexion des signaux reste la même
        self.checkbox_show_x_ticks.stateChanged.connect(self.on_show_x_ticks_changed)
        self.line_edit_x_ticks.editingFinished.connect(self.on_x_ticks_edited)
        self.line_edit_x_ticks.returnPressed.connect(self.on_x_ticks_edited)
        
        self.checkbox_show_y_ticks.stateChanged.connect(self.on_show_y_ticks_changed)
        self.line_edit_y_ticks.editingFinished.connect(self.on_y_ticks_edited)
        self.line_edit_y_ticks.returnPressed.connect(self.on_y_ticks_edited)
        
        options_layout.addStretch() # Pousse tout vers le haut
        return options_widget

    def populate_listFile_widget(self):
        """Remplit le QListWidget avec les chaînes de caractères de gpr_data_array."""
        self.listFile_widget.clear() # S'assurer que la liste est vide avant de la remplir
        
        for file in self.basalt.getFilesFiltered(self.constante.getFiltreFreq(self.combo_box_frequence.currentText()),
                        self.basalt.getFilesInFolder(self.constante.getFiltreExtension(self.combo_box_extension.currentText()))):
            list_item = QListWidgetItem(file.stem)
            self.listFile_widget.addItem(list_item)
    
    def _parse_input_to_float(self, text: str, default_on_error: float = 0.0, return_none_if_empty: bool = False) -> Union[float, None]:
        """
        Convertit un texte en float, en gérant la virgule et les erreurs.
        
        Args:
            text (str): Le texte à convertir.
            default_on_error (float): Valeur à retourner si le texte n'est pas un nombre valide.
            return_none_if_empty (bool): Si True, retourne None si le texte est vide. Sinon, retourne default_on_error.
        """
        if not text:
            # Si le champ est vide, on retourne soit None, soit la valeur par défaut
            return None if return_none_if_empty else default_on_error
        
        try:
            # Remplace la virgule par un point et tente la conversion
            return float(text.replace(',', '.'))
        except ValueError:
            # En cas d'autre erreur (ex: "abc"), retourne la valeur par défaut
            return default_on_error
    
    def on_file_clicked(self, item):
        """Gère le clic sur un élément de la liste."""
        print(f"Élément sélectionné: {item.text()}")

        for file in self.basalt.getFilesFiltered(self.constante.getFiltreFreq(self.combo_box_frequence.currentText()),
                    self.basalt.getFilesInFolder(self.constante.getFiltreExtension(self.combo_box_extension.currentText()))):
            if file.stem == item.text():
                # 1. Charger le fichier (ceci remplit les bonnes valeurs dans Basalt)
                self.basalt.setSelectedFile(file, self.constante.getRadarByExtension(self.combo_box_extension.currentText()))
                header = self.basalt.data.header

                # Mettre à jour les fréquences
                self.line_edit_freq_filtre.setText(f"{header.sampling_frequency / 1e6:.2f}")
                self.line_edit_antenna_freq.setText(f"{header.antenna_frequency:.1f}")

                # Mettre à jour les placeholders des limites pour guider l'utilisateur
                self.line_edit_xlim.setPlaceholderText(f"Max: {header.value_trace}")
                self.line_edit_ylim.setPlaceholderText(f"Max: {header.value_sample}")

                # 3. Forcer la re-lecture des valeurs des champs de limites actuels
                # En appelant les handlers, on s'assure que les valeurs de l'UI
                # sont bien celles utilisées pour le traitement.
                self.on_x0_edited()
                self.on_xlim_edited()
                self.on_y0_edited()
                self.on_ylim_edited() # Cet appel va déclencher le update_display final
                return
        return None
    def on_mouse_move(self, event):
        """
        Gère les événements de mouvement de la souris sur le canvas Matplotlib.
        """
        # L'événement contient les coordonnées en pixels (event.x, event.y)
        # et en données (event.xdata, event.ydata)

        # D'abord, on vérifie si la souris est bien à l'intérieur de nos axes de graphique
        if event.inaxes is self.ax:
            # event.xdata et event.ydata nous donnent directement les coordonnées
            # dans le système de l'axe (mètres, ns, samples, etc.)
            x_coord = event.xdata
            y_coord = event.ydata

            # On formate le texte pour l'affichage avec 2 décimales
            coord_text = f"X: {x_coord:.2f} | Y: {y_coord:.2f}"
            
            # On met à jour le texte de notre label
            self.coord_label.setText(coord_text)
        else:
            # Si la souris sort des axes, on efface les coordonnées
            self.coord_label.setText("X: -- | Y: --")
    def on_contrast_slider_changed(self, value):
        """Gère le changement de valeur du slider de contraste."""
        real_contrast_value = value / 100.0
        self.current_contrast_label.setText(f"{real_contrast_value:.2f}")
        
        # On met à jour la valeur dans notre modèle de données
        self.basalt.traitementValues.contraste = real_contrast_value
        
        # On appelle la fonction LÉGÈRE qui ne fait que redessiner
        self.redraw_plot()
    
    def on_unit_x_changed(self, selected_unit_text):
        """Met à jour l'unité de l'axe X et rafraîchit l'affichage."""
        if self.basalt.data is None: return # Ne rien faire si aucun fichier n'est chargé
        self.basalt.traitementValues.unit_x = selected_unit_text
        self.update_display()

    def on_unit_y_changed(self, selected_unit_text):
        """Met à jour l'unité de l'axe Y et rafraîchit l'affichage."""
        if self.basalt.data is None: return
        self.basalt.traitementValues.unit_y = selected_unit_text
        self.update_display()
    
    def on_distance_edited(self):
        valeur = self._parse_input_to_float(self.line_edit_distance.text(), return_none_if_empty=True)
        
        self.basalt.traitementValues.X_dist = valeur
        print(f"Distance forcée mise à jour : {valeur}")
        self.update_display()

    def on_epsilon_edited(self):
        self.basalt.traitementValues.epsilon = self._parse_input_to_float(self.line_edit_epsilon.text(), default_on_error="")
        self.update_display()

    def on_x0_edited(self):
        self.basalt.traitementValues.X0 = self._parse_input_to_float(self.line_edit_x0.text())
        self.update_display()

    def on_xlim_edited(self):
        self.basalt.traitementValues.X_lim = self._parse_input_to_float(self.line_edit_xlim.text(), return_none_if_empty=True)
        self.update_display()

    def on_y0_edited(self):
        self.basalt.traitementValues.T0 = self._parse_input_to_float(self.line_edit_y0.text())
        self.update_display()

    def on_ylim_edited(self):
        self.basalt.traitementValues.T_lim = self._parse_input_to_float(self.line_edit_ylim.text(), return_none_if_empty=True)
        self.update_display()

    def on_trace_moyenne_edited(self): # Attention prend pas en charge distance / Sample
        self.basalt.traitementValues.sub_mean = self._parse_input_to_float(self.line_edit_trace_moyenne.text(), default_on_error="")
        self.update_display()

    def on_freq_filtre_edited(self):
        self.basalt.traitementValues.sampling_freq = self._parse_input_to_float(self.line_edit_freq_filtre.text(), default_on_error="")
        self.update_display()

    def on_cuttoff_freq_edited(self):
        self.basalt.traitementValues.antenna_freq =  self._parse_input_to_float(self.line_edit_antenna_freq.text(), default_on_error="")
        self.update_display()

    def on_dewow_toggled(self, state):
        """Gère le cochage/décochage de la case du dewow."""
        # 1. Détermine si la case est cochée
        is_checked = (state == Qt.CheckState.Checked.value)
        
        # 2. Met à jour la valeur dans notre modèle de données
        self.basalt.traitementValues.is_dewow_active = is_checked
        print(f"Dewow activé : {is_checked}")
        
        # 3. Relance un traitement complet pour appliquer ou retirer le dewow
        self.update_display()

    def on_sub_mean_toggled(self, state):
        is_checked = (state == Qt.CheckState.Checked.value)
        self.basalt.traitementValues.is_sub_mean = is_checked
        
        # Activer/désactiver les options en fonction de la case
        self.combo_sub_mean_mode.setEnabled(is_checked)
        # On active le champ de la fenêtre seulement si le mode est 'Mobile'
        is_mobile_mode = (self.combo_sub_mean_mode.currentText() == 'Mobile')
        self.line_edit_sub_mean_window.setEnabled(is_checked and is_mobile_mode)
        
        self.update_display()

    def on_sub_mean_mode_changed(self, index):
        mode_text = self.combo_sub_mean_mode.currentText()
        self.basalt.traitementValues.sub_mean_mode = mode_text
        
        # Activer/désactiver le champ de la taille de la fenêtre
        is_mobile_mode = (mode_text == 'Mobile')
        self.line_edit_sub_mean_window.setEnabled(is_mobile_mode)
        
        # Mettre à jour si la fonction est active
        if self.basalt.traitementValues.is_sub_mean:
            self.update_display()

    def on_sub_mean_window_edited(self):
        valeur = int(self._parse_input_to_float(self.line_edit_sub_mean_window.text(), default_on_error=21))
        self.basalt.traitementValues.sub_mean_window = valeur

        # Mettre à jour si la fonction est active
        if self.basalt.traitementValues.is_sub_mean:
            self.update_display()

    def on_filtre_freq_changed(self, state):
        """Gère le cochage/décochage de la case du filtre fréquentiel."""
        # Le signal stateChanged envoie un entier. On le convertit en booléen.
        is_checked = (state == Qt.CheckState.Checked.value)
        
        # On met à jour la valeur dans notre dataclass
        self.basalt.traitementValues.is_filtre_freq = is_checked
        print(f"Filtre fréquentiel activé : {is_checked}")
        
        # On relance un traitement complet pour appliquer/retirer le filtre
        self.update_display()
        
    def on_g_edited(self):
        self.basalt.traitementValues.g = self._parse_input_to_float(self.line_edit_g.text(), default_on_error=1.0)
        self.update_display()
    def on_a_lin_edited(self):
        self.basalt.traitementValues.a_lin = self._parse_input_to_float(self.line_edit_a_lin.text(), default_on_error=0.0) / 100
        self.update_display()
    def on_t0_lin_edited(self):
        self.basalt.traitementValues.t0_lin = self._parse_input_to_float(self.line_edit_t0_lin.text(), default_on_error=0.0)
        self.update_display()
    def on_a_exp_edited(self):
        self.basalt.traitementValues.a_exp = self._parse_input_to_float(self.line_edit_a_exp.text(), default_on_error=0.0)
        self.update_display()
    def on_T0_exp_edited(self):
        self.basalt.traitementValues.t0_exp = self._parse_input_to_float(self.line_edit_T0_exp.text(), default_on_error=0.0)
        self.update_display()
    def on_t_max_exp_edited(self):
        """Gère la modification de la limite du gain exponentiel."""
        # On utilise int() car c'est un numéro d'échantillon
        valeur = int(self._parse_input_to_float(self.line_edit_t_max_exp.text(), default_on_error=0))
        self.basalt.traitementValues.t_max_exp = valeur
        self.update_display()

    def on_show_x_ticks_changed(self, state):
        is_checked = (state == Qt.CheckState.Checked.value)
        self.basalt.traitementValues.show_x_ticks = is_checked
        self.update_display()

    def on_x_ticks_edited(self):
        valeur = int(self._parse_input_to_float(self.line_edit_x_ticks.text(), default_on_error=20))
        self.basalt.traitementValues.X_ticks = valeur
        self.update_display()

    def on_show_y_ticks_changed(self, state):
        is_checked = (state == Qt.CheckState.Checked.value)
        self.basalt.traitementValues.show_y_ticks = is_checked
        self.update_display()

    def on_y_ticks_edited(self):
        valeur = int(self._parse_input_to_float(self.line_edit_y_ticks.text(), default_on_error=20))
        self.basalt.traitementValues.Y_ticks = valeur
        self.update_display()

    def open_folder(self):
        selected_folder = QFileDialog.getExistingDirectory(self.window, "Ouvrir un dossier")
        self.basalt.setFolder(selected_folder)
        self.populate_listFile_widget()
        
    def exportPNG(self):
        """
        Sauvegarde le radargramme actuellement affiché dans un fichier image (PNG, JPG).
        """
        # 1. Vérifier qu'un fichier est bien chargé
        if self.basalt.selectedFile is None:
            print("Aucun fichier à sauvegarder.")
            # On pourrait afficher un QMessageBox pour l'utilisateur ici
            return

        # 2. Proposer un nom de fichier par défaut (ex: C:/.../scan1_traite.png)
        # Vous aviez déjà une méthode pour ça, mais construisons-le ici pour plus de clarté
        default_folder = self.basalt.folder
        default_filename = self.basalt.selectedFile.stem + "_traite.png"
        default_path = os.path.join(default_folder, default_filename)
        
        # 3. Ouvrir la boîte de dialogue "Enregistrer sous..."
        fileName, _ = QFileDialog.getSaveFileName(self.window, 
                                                "Sauvegarder l'image", 
                                                default_path, 
                                                "Images PNG (*.png);;Images JPEG (*.jpg);;Tous les fichiers (*)")

        # 4. Si l'utilisateur a bien choisi un nom de fichier (n'a pas annulé)
        if fileName:
            try:
                # On utilise la méthode savefig de la figure Matplotlib
                # dpi=300 pour une bonne résolution d'impression
                self.fig.savefig(fileName, dpi=300)
                print(f"Image sauvegardée avec succès sous : {fileName}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde de l'image : {e}")
                # Ici aussi, un QMessageBox serait idéal pour notifier l'utilisateur de l'erreur


    def exportPNG_all(self):
        """
        Traite et sauvegarde tous les scans de la liste de fichiers actuelle,
        en utilisant les paramètres de traitement courants.
        """
        # 1. Vérifier qu'il y a des fichiers à exporter
        if self.listFile_widget.count() == 0:
            print("Aucun fichier dans la liste à exporter.")
            return

        # 2. Demander à l'utilisateur de choisir un dossier de destination
        output_folder = QFileDialog.getExistingDirectory(self.window, "Choisir un dossier de destination pour l'export")

        if not output_folder:
            print("Export annulé.")
            return

        print(f"Début de l'export vers le dossier : {output_folder}")

        # 3. Boucler sur tous les éléments de la QListWidget
        for i in range(self.listFile_widget.count()):
            item = self.listFile_widget.item(i)
            scan_name = item.text()
            
            print(f"Traitement du fichier {i+1}/{self.listFile_widget.count()} : {scan_name}")

            # 4. Simuler un clic sur le fichier pour le charger et le traiter
            # Trouvons le chemin complet du fichier
            # Note: cette recherche peut être optimisée, mais elle est robuste
            file_path_to_process = None
            current_files_in_list = self.basalt.getFilesFiltered(
                self.constante.getFiltreFreq(self.combo_box_frequence.currentText()),
                self.basalt.getFilesInFolder(self.constante.getFiltreExtension(self.combo_box_extension.currentText()))
            )
            for file in current_files_in_list:
                if file.stem == scan_name:
                    file_path_to_process = file
                    break
            
            if file_path_to_process:
                # On charge et on traite le fichier avec les paramètres actuels
                self.basalt.setSelectedFile(file_path_to_process, self.constante.getRadarByExtension(file_path_to_process.suffix))
                self.update_display() # Cette fonction fait tout : traitement + mise à jour du graphique

                # 5. Construire le nom du fichier de sortie et sauvegarder
                output_filename = os.path.join(output_folder, scan_name + "_traite.png")
                try:
                    self.fig.savefig(output_filename, dpi=300)
                    print(f" -> Fichier sauvegardé : {output_filename}")
                except Exception as e:
                    print(f" -> ERREUR lors de la sauvegarde de {output_filename} : {e}")

            # 6. Ligne CRUCIALE pour garder l'interface réactive pendant la boucle
            QApplication.processEvents()

        print("--- Export de tous les scans terminé ! ---")
        # On pourrait afficher un QMessageBox pour confirmer la fin de l'export

    @property
    def getFiles(self): 
        files = self.basalt.getFilesInFolder('*.dzt')
        return self.basalt.getFilesFiltered(None,files)

class Basalt():
    def __init__(self):
        self.folder :str = ""
        self.data : RadarData = None
        self.traitement : Traitement = None
        self.selectedFile : Path = None
        
        self.antenna : Radar
        self.subMean : bool = False
        self.filtreFreq : bool = False
        self.dewow : bool = False
        self.traitementValues : TraitementValues = TraitementValues()

        self.boolFlex : bool = True

        self.epsilon :float = 8 

    def setFolder(self,folder:str):
        if os.path.isdir(folder):
            self.folder = folder
        else: 
            self.folder = None 
            
    def getFilesInFolder(self, filtre:str = "*"):
        """
            Renvoie la liste des fichiers d'un dossier avec une possibilité de filtre sur l'extension (optionnel)
            ex : filtre = "*.rd7"
        """
        if self.folder != None :
            return list(Path(self.folder).glob(filtre))
        else:
            return None
        
    def getFilesFiltered(self, key:str = None, files = []):
        """
            Permet de filtrer la liste des fichiers dans le dossier selon une clef de recherche
            Ex : _1 en key pour chercher la haute fréquence de Mala
        """
        if key is None : return files
        return [f for f in files if Path(f).stem.endswith(key)]

    
    def setSelectedFile(self, GPR_File:Path, radar : Radar):

        self.antenna = radar
        self.selectedFile = GPR_File
        self.data = RadarData(GPR_File,radar)

        # On garde la détection automatique des fréquences
        fs_hz = self.data.header.sampling_frequency
        self.traitementValues.sampling_freq = fs_hz
        print(f"Fréquence d'échantillonnage détectée : {fs_hz / 1e6:.2f} MHz")

        ant_freq_mhz = self.data.header.antenna_frequency
        self.traitementValues.antenna_freq = ant_freq_mhz
        print(f"Fréquence d'antenne détectée : {ant_freq_mhz} MHz")

    def traitementScan(self): ## problème : a chaque changement besoin de repasser par ici (traitement, epsilon, tout ça tout ça)
        """
            Applique tous les gains, filtre, et cut sur le tableau brute
        """
        self.traitement = Traitement(self.getTableCuté(self.data.dataFile),
                                 y_offset=self.traitementValues.T0)
        
        if self.dewow : 
            self.traitement.dewow_filter()

        if self.traitementValues.is_sub_mean:
            self.traitement.sub_mean(mode=self.traitementValues.sub_mean_mode,
                                    window_size=self.traitementValues.sub_mean_window)
        if self.traitementValues.is_filtre_freq : 
            self.traitement.filtre_frequence(self.traitementValues.antenna_freq, 
                                          self.traitementValues.sampling_freq)

        self.traitement.apply_total_gain(t0_exp = self.traitementValues.t0_exp,
                                          t0_lin = self.traitementValues.t0_lin,
                                          g= self.traitementValues.g ,
                                          a_lin= self.traitementValues.a_lin ,
                                          a = self.traitementValues.a_exp,
                                          t_max_exp=self.traitementValues.t_max_exp)

    def _getDataTraité(self,data): 
        if self.antenna is Radar.GSSI_FLEX: 
            return self._getFlexData(data)
        else:
            return data 

    def _getFlexData(self, data):
        # On détermine le point de césure au milieu des échantillons
        mid_point = data.shape[0] // 2  # Division entière, ex: 2048 // 2 = 1024

        if self.boolFlex: 
            # Premier canal
            return data[0:mid_point, :]
        else:
            # Deuxième canal
            return data[mid_point:, :]
        
    def getTableCuté(self,data):
        """
        Découpe le tableau de données en utilisant les limites définies dans l'UI.
        Cette version est sécurisée et garantit l'utilisation d'indices entiers.
        """
        # Dimensions maximales des données originales
        total_samples, total_traces = data.shape

        # --- SÉCURISATION ET CONVERSION DES LIMITES ---

        # On récupère les valeurs (qui peuvent être des floats ou None)
        t0_raw = self.traitementValues.T0
        t_lim_raw = self.traitementValues.T_lim
        x0_raw = self.traitementValues.X0
        x_lim_raw = self.traitementValues.X_lim

        # LA CORRECTION CLÉ : On convertit en ENTIERS et on gère les valeurs None
        y_start = int(t0_raw)
        y_end = int(t_lim_raw if t_lim_raw is not None else total_samples)
        x_start = int(x0_raw)
        x_end = int(x_lim_raw if x_lim_raw is not None else total_traces)

        # On s'assure que les indices sont valides (ni négatifs, ni hors-limites)
        y_start = max(0, y_start)
        y_end = min(total_samples, y_end)
        x_start = max(0, x_start)
        x_end = min(total_traces, x_end)

        # Sécurité supplémentaire : si les limites sont inversées, on retourne un tableau vide
        if y_start >= y_end or x_start >= x_end:
            print("Avertissement : Limites de découpage invalides (début >= fin).")
            return np.array([[]], dtype=data.dtype)

        print(f"Découpage des données (indices entiers) : Y de {y_start} à {y_end}, X de {x_start} à {x_end}")

        # On utilise ces indices entiers et sécurisés pour découper les données
        cropped_data = self._getDataTraité(data)[y_start:y_end, x_start:x_end]
        
        return cropped_data

    def get_plot_axes_parameters(self):
        """
        Calcule l'extent et les labels pour le graphique en fonction des unités choisies.
        """
        if self.data is None:
            return None, "", ""

        header = self.data.header
        y_start_sample = int(self.traitementValues.T0)
        y_end_sample = int(self.traitementValues.T_lim if self.traitementValues.T_lim is not None else header.value_sample)
        x_start_trace = int(self.traitementValues.X0)
        x_end_trace = int(self.traitementValues.X_lim if self.traitementValues.X_lim is not None else header.value_trace)

        # Logique pour la distance par trace (m_par_trace)
        distance_totale = header.value_dist_total
        
        # On vérifie si une distance a été forcée par l'utilisateur
        if self.traitementValues.X_dist is not None and self.traitementValues.X_dist > 0:
            distance_totale = self.traitementValues.X_dist
            print(f"Utilisation de la distance forcée : {distance_totale} m")
        
        m_par_trace = distance_totale / header.value_trace if header.value_trace else 0

        # Le reste des calculs de conversion est maintenant correct
        ns_par_sample = header.value_time / header.value_sample if header.value_sample else 0
        s_par_trace = header.value_step_time_acq if header.value_step_time_acq is not None else 0
        vitesse = cste_global["c_lum"] / sqrt(self.traitementValues.epsilon)

        # --- Calcul pour l'axe X ---
        unit_x = self.traitementValues.unit_x
        if unit_x == "Distance (m)":
            x_start, x_end = x_start_trace * m_par_trace, x_end_trace * m_par_trace
            xlabel = "Distance (m)"
        elif unit_x == "Temps (s)":
            x_start, x_end = x_start_trace * s_par_trace, x_end_trace * s_par_trace
            xlabel = "Temps d'acquisition (s)"
        else: # Pour "Traces"
            x_start, x_end = x_start_trace, x_end_trace
            xlabel = "N° de Trace"

        # --- Calcul pour l'axe Y ---
        unit_y = self.traitementValues.unit_y
        plage_samples = y_end_sample - y_start_sample
        
        if unit_y == "Profondeur (m)":
            y_start = 0
            plage_de_temps_ns = plage_samples * ns_par_sample
            y_end = (plage_de_temps_ns * 1e-9) * vitesse / 2
            ylabel =  f"Profondeur relative (m) ; ε = {self.traitementValues.epsilon}"
        elif unit_y == "Temps (ns)":
            y_start = 0
            y_end = plage_samples * ns_par_sample
            ylabel = "Temps relatif (ns)"
        else: # Pour "Samples"
            y_start = 0
            y_end = plage_samples
            ylabel = "N° d'Échantillon relatif"

        plot_extent = [x_start, x_end, y_end, y_start]
        
        return plot_extent, xlabel, ylabel

    @property
    def getExportName(self):
        return self.folder + "/" + self.selectedFile.stem + ".png"

class RadarData():
    """
        Données Brutes
    """
    def __init__(self, path :Path, radar: Radar):
        """
            path (str) : le chemin du fichier gpr
            pathHeader (str) : le chemin de rad et dxt
            radar (Radar) : le type d'appareil 
            dataFile : les données
            header : le Header -_-
        """
        self.path :Path = path
        self.header = self.Header()
        self.header.readHeader(path,radar)
        self.radar : Radar = radar

        self.dataFile = self.read_data()

    def read_data(self):
        """
    Méthode permettant de récupérer la zone sondée à partir d'un fichier

    Return:
        Retourne le tableau numpy contenant les données de la zone sondée.
        """
        if(self.path.suffix == ".rd3"):
            # Ouvrir le fichier en mode binaire "rb"
            with open(self.path, mode='rb') as rd3data:  
                byte_data = rd3data.read()
            # rd3 est codé sur 2 octets
            rd3 = np.frombuffer(byte_data, dtype=np.int16) 
            # Reshape de rd3
            rd3 = rd3.reshape(self.header.value_trace, self.header.value_sample) 
            rd3 = rd3.transpose()
            return rd3
        
        elif(self.path.suffix == ".rd7"):
            # Ouvrir le fichier en mode binaire "rb"
            with open(self.path, mode='rb') as rd7data:  
                byte_data = rd7data.read()
                # rd7 est codé 4 octets
            rd7 = np.frombuffer(byte_data, dtype=np.int32)
            # Reshape de rd7
            rd7 = rd7.reshape(self.header.value_trace, self.header.value_sample)
            rd7 = rd7.transpose()
            return rd7
        
        elif(self.path.suffix == ".DZT"):
            # Ouvrir le fichier en mode binaire
            with open(self.path, mode='rb') as DZTdata:
                byte_data = DZTdata.read()
                # DZT est codé 4 octets
            DZT = np.frombuffer(byte_data, dtype=np.int32)[(2**15):,]
            # Reshape de rd7
            DZT = DZT.reshape(self.header.value_trace, self.header.value_sample)
            DZT = DZT.transpose()
            return DZT
        
        elif(self.path.suffix == ".dzt"): #Flex 
            # Ouvrir le fichier en mode binaire
            with open(self.path, mode='rb') as DZTdata:
                byte_data = DZTdata.read()
                # DZT est codé 4 octets
            DZT = np.frombuffer(byte_data, dtype=np.int32)[(2**15):,]
            # Reshape de rd7
            DZT = DZT.reshape(self.header.value_trace, self.header.value_sample)
            DZT = DZT.transpose()
            return DZT            
            # À supprimer
            #README
            # Si vous souhaitez rajouter d'autres format:
            # -1 Ajouter elif(self.path.endswith(".votre_format")):
            # -2 Veuillez vous renseigner sur la nature de vos données binaire, héxadécimal ...
            # -3 Lire les fichiers et ensuite les transférer dans un tableau numpy
            # -4 Redimmensionnez votre tableau à l'aide du nombre de samples et de traces
            # -5 Transposez le tableau
    
    class Header():
        """
            Infos générique du fichier (.rad, dxt, ou en-tête)

            Return:
                Retourne les informations suivantes :\n
                    - trace (int) : nombre de mesures\n
                    - samples (int): nombre d'échantillons\n
                    - distance total (float): distance totale\n
                    - time (float): Temps d'aller\n
                    - step (float): distance par mesure (horizontal (sol))\n
                    - stem time (float): temps par mesure (horizontal (sol))
        """
        def __init__(self):
            self.value_trace = None
            self.value_sample = None
            self.value_dist_total = None
            self.value_time = None
            self.value_step = None
            self.value_step_time_acq = None
            self.hdr = None

        @property
        def sampling_frequency(self) -> float:
            """
            Calcule et retourne la fréquence d'échantillonnage (fs) en Hz.
            Formule: fs = nombre_d_echantillons / fenetre_temporelle_en_secondes
            """
            if self.value_sample is None or self.value_time is None or self.value_time == 0:
                return 0.0

            # La fenêtre temporelle est souvent en nanosecondes (ns), il faut la convertir en secondes (* 1e-9)
            time_window_sec = self.value_time * 1e-9

            fs = self.value_sample / time_window_sec
            return fs
        @property
        def antenna_frequency(self) -> float:
            """
            Tente d'extraire la fréquence de l'antenne (en MHz) depuis le nom de l'antenne.
            """
            if not self.value_antenna:
                return 0.0 # Retourne 0 si le nom n'est pas trouvé

            # Cherche un nombre suivi de "mhz" ou "m"
            match = re.search(r'(\d+)\s*(mhz|m)', self.value_antenna.lower())
            if match:
                return float(match.group(1))
            return 500.0 # Retourne une valeur par défaut si non trouvé
      
        def readHeader(self, file : Path, radar :Radar):
            if radar == Radar.MALA :
                path = str(file.resolve().with_suffix(".rad"))
                
                # Lecture du fichier .rad
                with open(path, 'r') as file:
                    lines = file.readlines()

                # Traitement des lignes du fichier
                for line in lines:
                    # Supprimer les espaces en début et fin de ligne
                    line = line.strip()
                    if "SAMPLES" in line:
                        value = line.split(':')[1]
                        self.value_sample = int(value)
                    elif "LAST TRACE" in line:
                        value = line.split(':')[1]
                        self.value_trace = int(value)
                    elif "STOP POSITION" in line:
                        value = line.split(':')[1]
                        self.value_dist_total = float(value)
                    elif "TIMEWINDOW" in line:
                        value = line.split(':')[1]
                        self.value_time = float(value)
                    elif "DISTANCE INTERVAL" in line:
                        value = line.split(':')[1]
                        self.value_step = float(value)
                    elif "TIME INTERVAL" in line:
                        value = line.split(':')[1]
                        self.value_step_time_acq = float(value)
                    elif "ANTENNAS" in line:
                        value = line.split(':')[1]
                        self.value_antenna = value           
            elif radar == Radar.GSSI_FLEX or radar == Radar.GSSI_XT:
                    self.hdr = dzt.readgssi(infile=file, zero=[0])[0]
                    self.value_trace = self.hdr['shape'][1]
                    self.value_sample = self.hdr['shape'][0]
                    self.value_dist_total = self.value_trace / self.hdr['dzt_spm']
                    self.value_time = self.hdr['rhf_range']
                    self.value_step = self.hdr['dzt_spm']
                    self.value_step_time_acq = self.hdr['dzt_sps']
                    self.value_antenna = self.hdr['rh_antname'][0]

class Traitement():
    """
        Données traités
    """
    def __init__(self, rawData, y_offset: int = 0):
        self.data = rawData
        # On mémorise le décalage
        self.y_offset = int(y_offset) 

    def apply_total_gain(self, t0_lin: int, t0_exp: int, g: float, a_lin: float, a: float, t_max_exp: int):
        """
        Méthode permettant d'appliquer le gain souhaité à l'image.
        Args:
                t0 (float): La ligne à partir de laquelle le gain doit être appliqué.
                g (float): Coefficient du gain normal
                a_lin (float): Coefficient du gain linéaire (Fonction linéaire f:x --> a(x-t0)
                a (float): Coefficient d'atténuation de l'exponentielle (Fonction exponentielle: f: x --> exp(a(x-t0)))

        Returns:
                ndarray : Retourne le tableau traité.
        """
        # Conversion de l'image en flottant pour effectuer les calculs
        image_float = None
        #try:
        bits = self.get_bit_img()
        image_float = self.data.astype("float"+str(bits))
        samples = image_float.shape[0]
        fgain = None
        if(bits == 16):
            fgain = np.ones(samples, dtype=np.float16)
        else:
            if(bits == 32):
                fgain = np.ones(samples, dtype=np.float32)
            else:
                if(bits == 64):
                    fgain = np.ones(samples, dtype=np.float64)
                    print("Attention 64 bits")
                else:
                    print(f"Erreur: bits{bits}")
        L = np.arange(self.y_offset, self.y_offset + samples)
        
        # Gain constant
        fgain *= g

        # Gain linéaire
        b_lin = 1 - a_lin * t0_lin
        # On applique le gain sur les samples qui sont APRÈS t0_lin dans le temps ABSOLU
        gain_mask = L >= t0_lin 
        fgain[gain_mask] += a_lin * L[gain_mask] + b_lin

        # Gain exponentiel

        a = 1 + a / 10
        if a > 1: # Simplification
            b = np.log(a) / 75 
            
            # Le décalage initial pour l'exponentiel doit aussi être en temps absolu
            exp_start_sample = self.y_offset + int(t0_exp)

            # La fin de la plage est aussi en temps absolu
            end_range_sample = self.y_offset + (samples if (t_max_exp <= t0_exp or t_max_exp >= samples) else t_max_exp)
            
            # Masque pour la partie exponentielle
            exp_mask = (L >= exp_start_sample) & (L < end_range_sample)
            fgain[exp_mask] += a * (np.exp(b * (L[exp_mask] - exp_start_sample)))

            # Masque pour la partie constante après la fin de l'exponentiel
            flat_mask = L >= end_range_sample
            if np.any(flat_mask) and np.any(exp_mask):
                last_gain_value = fgain[exp_mask][-1] # La dernière valeur calculée
                fgain[flat_mask] = last_gain_value
        


        image_avec_gain = image_float * fgain[:, np.newaxis]

        # On s'assure que les valeurs ne dépassent pas les limites du type de données original
        val_min = -(2**(bits-1)) # Correction: utiliser bits-1 pour les entiers signés
        val_max = (2**(bits-1)) - 1
        
        # Appliquer le clip SUR L'IMAGE AVEC GAIN
        self.data = np.clip(image_avec_gain, val_min, val_max)
        
        # On reconvertit au type entier original pour garder la cohérence
        self.data = self.data.astype(image_float.dtype)

        #except:
        #    print("Erreur lors de l'application des gains:")
    
    def get_bit_img(self):
        """
        Récupère le nombre de bits à partir d'un tableau (néccessaire pour les fonctions gain).

        Args:
            img (numpy.ndarray): L'image d'entrée sous forme de tableau NumPy.

        Returns:
            int: Le nombre de bits du format.
        """
        #try:
        format = self.data.dtype.name
        if format.startswith("float"):
            return int(format[5:])
        else:
            if(format.startswith("int")):
                return int(format[3:])
        #except:
        #    print("Erreur lors de la lecture des bits:")
        
    def dewow_filter(self, window_size: int = 31):
        """
        Applique un filtre dewow en soustrayant une moyenne mobile sur chaque trace (verticalement).
        Version optimisée.
        """
        print(f"Application du filtre Dewow avec une fenêtre de {window_size}...")
        # On assure une fenêtre impaire
        if window_size % 2 == 0:
            window_size += 1
            
        data_float = self.data.astype(np.float32)
        
        # Calcul de la moyenne mobile sur l'axe du temps/des échantillons (axis=0)
        moving_average = uniform_filter1d(data_float, size=window_size, axis=0, mode='nearest')
        
        self.data = (data_float - moving_average).astype(self.data.dtype)

    def sub_mean(self, mode: str, window_size: int):
        """
        Soustrait la trace moyenne en mode 'Globale' ou 'Mobile'.
        Version optimisée sans boucle for.

        Args:
            mode (str): 'Globale' ou 'Mobile'.
            window_size (int): Taille de la fenêtre pour le mode mobile (doit être un entier impair).
        """
        print(f"Application du retrait de trace moyenne en mode '{mode}'...")
        
        # On travaille sur une copie en flottant
        data_float = self.data.astype(np.float32)

        if mode == 'Globale':
            # Calcul de la moyenne de toutes les traces, en une seule ligne
            mean_trace = np.mean(data_float, axis=1, keepdims=True)
            # Soustraction de la moyenne à toutes les traces en une seule opération (broadcasting)
            self.data = (data_float - mean_trace).astype(self.data.dtype)
            
        elif mode == 'Mobile':
            # Assurer que la taille de la fenêtre est un entier impair
            if window_size % 2 == 0:
                window_size += 1
            
            # Calcul de la moyenne mobile sur l'axe des traces (axis=1) en une seule ligne
            # mode='nearest' gère bien les bords de l'image
            moving_average = uniform_filter1d(data_float, size=window_size, axis=1, mode='nearest')
            
            # Soustraction de la moyenne mobile
            self.data = (data_float - moving_average).astype(self.data.dtype)


    def filtre_frequence(self, antenna_freq: float, sampling_freq: float):
        try:
            nyquist_hz = sampling_freq / 2.0
            
            low_cutoff_mhz = antenna_freq * 0.25
            high_cutoff_mhz = antenna_freq * 2.0
            if low_cutoff_mhz >= high_cutoff_mhz:
                print("Erreur : Fréquence basse >= Fréquence haute. Filtre non appliqué.")
                return
            
            # Normalisation des fréquences de coupure
            low_norm = (low_cutoff_mhz * 1e6) / nyquist_hz
            high_norm = (high_cutoff_mhz * 1e6) / nyquist_hz

            # Vérification des limites
            if high_norm >= 1.0: high_norm = 0.99
            if low_norm <= 0.0: low_norm = 0.01

            if low_norm >= high_norm:
                print("Erreur de normalisation des fréquences. Filtre non appliqué.")
                return
            # Création d'un filtre passe-bande de Butterworth
            order = 4
            b, a = signal.butter(order, [low_norm, high_norm], btype='band', analog=False)

            # Application du filtre sans déphasage
            self.data = signal.filtfilt(b, a, self.data, axis=0).astype(self.data.dtype)
        except Exception as e:
            print(f"Erreur lors de l'application du filtre passe-bande: {e}")

class Graphique():
    def __init__(self, ax : plt.Axes, fig : Figure):
        self.vmin = -5e9
        self.vmax = 5e9
        self.contraste = 1.0
        self.fig : Figure = fig
        self.ax : plt.Axes = ax
        self.im = None

    def setVerticalgrid(self,flag : bool):
        self.ax.grid(visible=flag, axis='x',linewidth = 0.5, color = "black", linestyle ='-.')

    def setHorizontalgrid(self,flag : bool):
        self.ax.grid(visible=flag, axis='y',linewidth = 0.5, color = "black", linestyle ='-.')

    def plot(self, data, title:str, x_ticks: int, y_ticks: int, extent: list, xlabel: str, ylabel: str, show_x_ticks: bool, show_y_ticks: bool):
        titre_complet = "Scan : " + title
        
        if self.im is None:
            self.im = self.ax.imshow(data, cmap="gray", aspect="auto", interpolation='nearest', extent=extent)
            self.fig.suptitle(titre_complet, y=0.05, fontsize=12)
            self.ax.xaxis.set_label_position('top')
            self.ax.xaxis.set_ticks_position('top')
        else:
            self.im.set_data(data)
            self.im.set_extent(extent)
            self.fig.suptitle(titre_complet, y=0.05, fontsize=12)

        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        
        # On met à jour les labels à chaque fois
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # Le reste est inchangé
        vmin, vmax = self.getRangePlot()
        self.im.set_clim(vmin=vmin, vmax=vmax)

        self.ax.locator_params(axis='x', nbins=x_ticks)
        self.ax.locator_params(axis='y', nbins=y_ticks)

    # --- GESTION DE LA GRILLE (Version Définitive) ---

        # 1. On efface TOUTES les grilles précédentes pour repartir de zéro.
        self.ax.grid(False)

        # 2. On redessine la grille X (verticale) UNIQUEMENT si la checkbox est cochée.
        if show_x_ticks:
            self.ax.grid(True, axis='x', color='black', linestyle='-.', linewidth=0.5)

        # 3. On redessine la grille Y (horizontale) UNIQUEMENT si la checkbox est cochée.
        if show_y_ticks:
            self.ax.grid(True, axis='y', color='black', linestyle='-.', linewidth=0.5)

        # --- FIN DE LA CORRECTION ---

        self.ax.figure.canvas.draw()

    def getRangePlot(self):
        """
        Calcule vmin et vmax en appliquant le contraste (multiplicateur)
        à une plage de valeurs de base fixes.
        """
        # self.contraste est la valeur du slider (ex: de 0.01 à 1.00)
        # self.vmin et self.vmax sont vos valeurs de base fixes.
        
        # On applique le contraste sur la plage de base.
        # Note : Si vous voulez que le slider ait plus ou moins d'effet,
        # vous pouvez changer la formule (ex: q = self.contraste / 10)
        # mais q = self.contraste est un bon point de départ.
        q = self.contraste 

        min_val = self.vmin * q
        max_val = self.vmax * q

        return min_val, max_val


if __name__ == '__main__':
    software_name = "Basalt - Le radar en profondeur"
    main_window = MainWindow(software_name)
    main_window.show()
