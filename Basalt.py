import numpy as np
import os
import readgssi.readgssi as dzt
import re
import numba

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from scipy import signal
from math import sqrt, floor
from scipy.ndimage import uniform_filter1d
from scipy.stats import trim_mean

import sys
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QListWidget, QRadioButton, QComboBox, QLineEdit, QTabWidget, QCheckBox, QSlider, QListWidgetItem, QGroupBox, QSplitter, QPushButton
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtGui import QDoubleValidator, QValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from typing import Union

ORGANIZATION_NAME = "HIF"
APPLICATION_NAME = "BasaltGPR"

cste_global = {
    "c_lum": 299792458, # Vitesse de la lumière dans le vide en m/s
    }

@numba.jit(nopython=True, cache=True)
def _kirchhoff_migration_loop(data, migrated_data, nt, nx, dt, dx, v, aperture_traces, y_offset):
    """
    La boucle de migration principale, optimisée par Numba.
    """
    # Boucle sur chaque pixel de l'image de SORTIE (l'image migrée)
    for i_x in range(nx):  # Pour chaque trace de sortie
        # On peut ignorer le premier échantillon de sortie s'il est à t=0
        for i_t in range(1, nt):  # Pour chaque échantillon de temps/profondeur
            
            # Temps (two-way) du point de sortie
            t0 = i_t * dt
            # Position x du point de sortie
            x = i_x * dx
            
            # Profondeur z du point de sortie
            z = v * t0 / 2.0
            
            sum_val = 0.0
            
            # On définit une "ouverture" pour limiter le calcul
            start_trace = max(0, i_x - aperture_traces)
            end_trace = min(nx, i_x + aperture_traces)

            # Boucle sur les traces de l'image d'ENTRÉE (le radargramme) dans l'ouverture
            for j_x in range(start_trace, end_trace):
                xj = j_x * dx
                
                # Calcul du temps de parcours (équation de l'hyperbole de diffraction)
                dist = np.sqrt((xj - x)**2 + z**2)
                t_hyperbole = (2.0 * dist) / v
                
                # On convertit ce temps en indice de sample ABSOLU
                i_t_in_absolu = int(round(t_hyperbole / dt))
                
                # On traduit l'indice absolu en indice LOCAL pour notre tableau découpé
                i_t_local = i_t_in_absolu - y_offset
                
                # On vérifie que cet indice LOCAL est valide dans notre tableau découpé
                if 0 <= i_t_local < nt:
                    sum_val += data[i_t_local, j_x]

            migrated_data[i_t, i_x] = sum_val

    return migrated_data

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
    sub_mean_trim_percent: float = 5.0 # Pourcentage à retirer de chaque côté (5% par défaut)

    
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
    contraste :float= 0.5

    unit_x: str = "Distance (m)"
    unit_y: str = "Profondeur (m)"

    profile_direction_mode: str = 'normal' # Options: 'normal', 'mirror_all', 'mirror_serpentine'
    interpolation_mode: str = 'nearest'
 
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
    def getFiltreFreq(self, freq: str):
        match freq:
            case "Haute Fréquence":
                return "_1"
            case "Basse Fréquence":
                return '_2'
            # Pour "Filtrage désactivé", "Canal 1", "Canal 2", etc., on ne filtre pas par nom de fichier.
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
        
        self.window.setGeometry(100, 100, 1600, 900)
        settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)

        geometry = settings.value("geometry")
        if geometry:
            self.window.restoreGeometry(geometry)

        last_folder = settings.value("last_folder", "") # "" est la valeur par défaut
        self.basalt.last_used_folder = last_folder

        # UI 
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

        # 1. On crée un QSplitter horizontal. C'est notre nouveau conteneur.
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # 2. On crée les graphiques comme avant
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.radargramme = Graphique(self.ax, self.fig)

        # A-Scan
        self.trace_fig = Figure(figsize=(4, 10), dpi=100)
        # On applique tight_layout à cette figure pour optimiser l'espace
        self.trace_fig.tight_layout() 
        self.trace_canvas = FigureCanvas(self.trace_fig)
        self.trace_ax = self.trace_fig.add_subplot(111)
        # On crée un objet Graphique dédié pour lui
        self.trace_plot = Graphique(self.trace_ax, self.trace_fig)

        # 3. On ajoute directement les canevas au splitter
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.trace_canvas)

        # 4. On peut toujours définir des proportions de départ
        # Le splitter les respectera au premier affichage.
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)

        # 5. On assemble la fenêtre principale
        self.main_layout.addWidget(self.control_panel_widget) # Bloc de gauche
        self.main_layout.addWidget(self.splitter)            # Bloc de droite (maintenant le splitter)

        self.window.closeEvent = self.closeEvent
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def closeEvent(self, event):
        """
        Cette méthode est automatiquement appelée par Qt juste avant que la fenêtre ne se ferme.
        C'est l'endroit parfait pour sauvegarder nos paramètres.
        """
        print("Fermeture de l'application, sauvegarde des paramètres...")
        settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
        
        # Sauvegarde de la géométrie de la fenêtre
        settings.setValue("geometry", self.window.saveGeometry())
        
        # Sauvegarde du dernier dossier utilisé
        settings.setValue("last_folder", self.basalt.last_used_folder)
        
        # On accepte l'événement de fermeture pour que l'application se ferme normalement
        event.accept()

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
        self.label_frequence = QLabel("Filtre fréquence :") 
        
        self.combo_box_frequence = QComboBox()
        self.combo_box_frequence.addItems(self.constante.freq_state)
        
        # On utilise la nouvelle variable
        frequence_layout.addWidget(self.label_frequence)
        frequence_layout.addWidget(self.combo_box_frequence)
        self.control_layout.addLayout(frequence_layout)


        # On connecte les signaux après avoir créé les objets
        self.combo_box_extension.currentTextChanged.connect(self.on_extension_changed)
        self.combo_box_frequence.currentTextChanged.connect(self.on_secondary_filter_changed)
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
        self.slider_contrast.setValue(50)     # Valeur initiale, correspond à 0.50
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
        self.line_edit_x0.returnPressed.connect(self.on_x0_edited)

        self.label_xlim = QLabel("X_lim:")
        self.xlim_layout.addWidget(self.label_xlim)
        self.line_edit_xlim = QLineEdit()
        self.line_edit_xlim.setPlaceholderText("Fin X")
        self.line_edit_xlim.setValidator(self.float_validator)
        self.xlim_layout.addWidget(self.line_edit_xlim)
        self.line_edit_xlim.editingFinished.connect(self.on_xlim_edited)
        self.line_edit_xlim.returnPressed.connect(self.on_xlim_edited)

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

        self.btn_auto_t0 = QPushButton("Auto")
        self.btn_auto_t0.setToolTip("Détecter automatiquement la première arrivée (T0)")
        self.ylim_layout.addWidget(self.btn_auto_t0)


        self.line_edit_y0.editingFinished.connect(self.on_y0_edited)
        self.line_edit_y0.returnPressed.connect(self.on_y0_edited)
        self.btn_auto_t0.clicked.connect(self.on_auto_t0_clicked)

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
                            show_y_ticks=self.basalt.traitementValues.show_y_ticks,
                            interpolation_mode=self.basalt.traitementValues.interpolation_mode)
                            
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

        # --- Retrait de la Trace Moyenne ---
        label_submean = QLabel("--- Retrait de la Trace Moyenne ---")
        label_submean.setStyleSheet("font-weight: bold; margin-top: 10px;")
        filter_layout.addWidget(label_submean)

        self.checkbox_sub_mean = QCheckBox("Activer le retrait de trace moyenne")
        filter_layout.addWidget(self.checkbox_sub_mean)

        # Layout pour les options qui vont changer dynamiquement
        sub_mean_options_layout = QHBoxLayout()
        # Widget pour les options du mode "Mobile"
        self.mobile_options_widget = QWidget()
        mobile_layout = QHBoxLayout(self.mobile_options_widget)
        mobile_layout.setContentsMargins(0,0,0,0)
        mobile_layout.addWidget(QLabel("Fenêtre :"))
        self.line_edit_sub_mean_window = QLineEdit(str(self.basalt.traitementValues.sub_mean_window))
        self.line_edit_sub_mean_window.setValidator(QDoubleValidator(1, 1001, 0))
        mobile_layout.addWidget(self.line_edit_sub_mean_window)

        # Widget pour les options du mode "Globale"
        self.globale_options_widget = QWidget()
        globale_layout = QHBoxLayout(self.globale_options_widget)
        globale_layout.setContentsMargins(0,0,0,0)
        globale_layout.addWidget(QLabel("Élagage (%) :"))
        self.line_edit_trim_percent = QLineEdit(str(self.basalt.traitementValues.sub_mean_trim_percent))
        self.line_edit_trim_percent.setValidator(QDoubleValidator(0, 49, 1)) # De 0% à 49%
        globale_layout.addWidget(self.line_edit_trim_percent)

        # On ajoute le choix du mode et les deux widgets d'options au layout
        sub_mean_options_layout.addWidget(QLabel("Mode :"))
        self.combo_sub_mean_mode = QComboBox()
        self.combo_sub_mean_mode.addItems(["Globale", "Mobile"])
        sub_mean_options_layout.addWidget(self.combo_sub_mean_mode)
        sub_mean_options_layout.addWidget(self.globale_options_widget)
        sub_mean_options_layout.addWidget(self.mobile_options_widget)
        sub_mean_options_layout.addStretch()

        filter_layout.addLayout(sub_mean_options_layout)

        # On initialise la visibilité et l'état "activé"
        self.checkbox_sub_mean.setChecked(False)
        self.combo_sub_mean_mode.setEnabled(False)
        self.globale_options_widget.setEnabled(False)
        self.mobile_options_widget.setEnabled(False)
        self.mobile_options_widget.setVisible(False) # Caché au départ car "Globale" est sélectionné

        # Connexion des signaux
        self.checkbox_sub_mean.stateChanged.connect(self.on_sub_mean_toggled)
        self.combo_sub_mean_mode.currentIndexChanged.connect(self.on_sub_mean_mode_changed)
        self.line_edit_sub_mean_window.editingFinished.connect(self.on_sub_mean_window_edited)
        self.line_edit_trim_percent.editingFinished.connect(self.on_trim_percent_edited) # <-- Nouvelle connexion

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

        filter_layout.addSpacing(20)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        filter_layout.addWidget(line)
        
        # Le bouton est maintenant ajouté directement au layout de la page
        self.btn_apply_migration = QPushButton("Appliquer la Migration Kirchhoff")
        self.btn_apply_migration.setToolTip("Attention : traitement lourd. Refocalise les hyperboles.")
        self.btn_apply_migration.clicked.connect(self.on_migration_clicked)
        filter_layout.addWidget(self.btn_apply_migration)

        # Connexion du signal
        self.btn_apply_migration.clicked.connect(self.on_migration_clicked)

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
        
        label_direction = QLabel("--- Sens du Profil ---")
        label_direction.setStyleSheet("font-weight: bold; margin-top: 20px;")
        options_layout.addWidget(label_direction)

        # On crée les boutons radio 
        self.radio_direction_normal = QRadioButton("Normal (par défaut)")
        self.radio_direction_normal.setChecked(True)
        self.radio_direction_mirror_all = QRadioButton("Inverser tous les profils (Miroir)")
        self.radio_direction_serpentine = QRadioButton("Inverser un profil sur deux (Serpentin)")
        
        # On les ajoute DIRECTEMENT au layout principal de l'onglet
        options_layout.addWidget(self.radio_direction_normal)
        options_layout.addWidget(self.radio_direction_mirror_all)
        options_layout.addWidget(self.radio_direction_serpentine)
        self.radio_direction_normal.toggled.connect(self.on_direction_mode_changed)
        self.radio_direction_mirror_all.toggled.connect(self.on_direction_mode_changed)
        self.radio_direction_serpentine.toggled.connect(self.on_direction_mode_changed)

        options_layout.addStretch() # Pousse tout vers le haut


        # --- Section pour la Qualité d'Affichage ---
        label_interpolation = QLabel("--- Qualité d'Affichage (Interpolation) ---")
        label_interpolation.setStyleSheet("font-weight: bold; margin-top: 10px;")
        options_layout.addWidget(label_interpolation)

        interpolation_layout = QHBoxLayout()
        interpolation_layout.addWidget(QLabel("Méthode :"))
        self.combo_interpolation = QComboBox()
        interpolation_options = ['nearest', 'bilinear', 'bicubic', 'lanczos', 'spline16']
        self.combo_interpolation.addItems(interpolation_options)
        interpolation_layout.addWidget(self.combo_interpolation)
        options_layout.addLayout(interpolation_layout)
        # -----------------

        return options_widget


    def populate_listFile_widget(self):
        """Remplit le QListWidget avec les noms de fichiers, triés numériquement."""
        self.listFile_widget.clear()

        # On fait un seul appel à notre nouvelle fonction intelligente
        sorted_files = self.basalt.getFilesFiltered(
            extension_filter=self.constante.getFiltreExtension(self.combo_box_extension.currentText()),
            freq_key=self.constante.getFiltreFreq(self.combo_box_frequence.currentText())
        )
        
        for file in sorted_files:
            list_item = QListWidgetItem(file.stem)
            self.listFile_widget.addItem(list_item)
    

    def on_extension_changed(self, extension_text: str):
        """
        Appelée quand l'extension change. Met à jour les options de la 2ème ComboBox.
        """
        # On bloque les signaux de la 2ème box pour éviter qu'elle ne se déclenche pendant qu'on la modifie
        self.combo_box_frequence.blockSignals(True)
        
        # On la vide de ses anciennes options
        self.combo_box_frequence.clear()

        # Si l'utilisateur a choisi un fichier GSSI Flex (.dzt)
        if extension_text == ".dzt":
            # --- MODIFICATION ICI ---
            # AVANT : self.label_combo_2.setText("Canal :")
            # APRÈS :
            self.label_frequence.setText("Canal :")
            self.combo_box_frequence.addItems(["Canal 1", "Canal 2"])
        # Pour tous les autres types de fichiers
        else:
            # --- MODIFICATION ICI ---
            # AVANT : self.label_combo_2.setText("Filtre fréquence :")
            # APRÈS :
            self.label_frequence.setText("Filtre fréquence :")
            self.combo_box_frequence.addItems(self.constante.freq_state)

        # On réactive les signaux
        self.combo_box_frequence.blockSignals(False)
        
        # On rafraîchit la liste des fichiers maintenant que les options de filtre sont à jour
        self.populate_listFile_widget()

    def on_secondary_filter_changed(self, selected_text: str):
        """
        Gère le changement de la 2ème ComboBox (qui peut être un filtre ou un sélecteur de canal).
        """
        extension = self.combo_box_extension.currentText()

        # Si on est en mode GSSI Flex
        if extension == '.dzt':
            # On met à jour la variable qui contrôle le canal à afficher
            self.basalt.boolFlex = (selected_text == "Canal 1")
            print(f"Changement de canal GSSI Flex. Affichage du canal {'1' if self.basalt.boolFlex else '2'}.")
            
            # Si un fichier est déjà chargé, on le retraite pour afficher le nouveau canal
            if self.basalt.data:
                self.update_display()
        # Sinon, on est en mode MALA
        else:
            # L'action est de rafraîchir la liste des fichiers avec le filtre _1 ou _2
            self.populate_listFile_widget()
        
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
        """
        Gère le clic sur un élément de la liste de manière robuste en utilisant l'index.
        """
        print(f"Élément sélectionné: {item.text()}")
        
        # 1. On récupère la liste de fichiers exactement comme elle est affichée
        current_files_in_list = self.basalt.getFilesFiltered(
            extension_filter=self.constante.getFiltreExtension(self.combo_box_extension.currentText()),
            freq_key=self.constante.getFiltreFreq(self.combo_box_frequence.currentText())
        )

        row = self.listFile_widget.row(item)
        
        # 3. On vérifie que cet index est valide et on récupère le bon fichier directement
        if 0 <= row < len(current_files_in_list):
            file_to_process = current_files_in_list[row]
        
            print(f"Fichier correspondant trouvé par index ({row}) : {file_to_process.name}")
            
            # 4. On charge le fichier en passant son index pour le mode serpentin
            self.basalt.setSelectedFile(file_to_process, self.constante.getRadarByExtension(file_to_process.suffix), row)
            
            # 5. On met à jour les champs de l'UI avec les infos du nouveau fichier
            header = self.basalt.data.header
            self.line_edit_freq_filtre.setText(f"{header.sampling_frequency / 1e6:.2f}")
            self.line_edit_antenna_freq.setText(f"{header.antenna_frequency:.1f}")
            self.line_edit_xlim.setPlaceholderText(f"Max: {header.value_trace}")
            self.line_edit_ylim.setPlaceholderText(f"Max: {header.value_sample}")

            # 6. On force la re-lecture des valeurs de l'UI pour les appliquer au nouveau fichier
            self.on_x0_edited()
            self.on_xlim_edited()
            self.on_y0_edited()
            self.on_ylim_edited() # Cet appel déclenche le update_display final
        else:
            print("Erreur : Impossible de retrouver le fichier correspondant à l'élément cliqué.")

    def on_mouse_move(self, event):
        """
        Gère les événements de mouvement de la souris, met à jour les coordonnées (X, Y)
        et l'amplitude (A), ainsi que le graphique de la trace.
        """
        # Si aucun fichier n'est chargé ou si la souris est en dehors des axes du radargramme
        if event.inaxes is not self.ax or self.basalt.traitement is None or self.basalt.traitement.data.size == 0:
            self.coord_label.setText("X: -- | Y: -- | A: --")
            return

        # 1. Récupération des coordonnées X et Y de la souris (inchangé)
        x_coord, y_coord = event.xdata, event.ydata
        
        # 2. Conversion des coordonnées en indices de tableau [ligne, colonne]
        data = self.basalt.traitement.data
        num_samples, num_traces = data.shape
        
        plot_extent, xlabel, ylabel = self.basalt.get_plot_axes_parameters()
        if plot_extent is None: return

        x_axis_start, x_axis_end = plot_extent[0], plot_extent[1]
        y_axis_top, y_axis_bottom = plot_extent[3], plot_extent[2] # Inversé: top=min, bottom=max

        # Calcul de l'indice de la trace (colonne)
        trace_idx = 0
        x_range = x_axis_end - x_axis_start
        if x_range > 0:
            relative_pos_x = (x_coord - x_axis_start) / x_range
            trace_idx = int(relative_pos_x * (num_traces - 1))

        # Calcul de l'indice du sample (ligne)
        sample_idx = 0
        y_range = y_axis_bottom - y_axis_top
        if y_range > 0:
            relative_pos_y = (y_coord - y_axis_top) / y_range
            sample_idx = int(relative_pos_y * (num_samples - 1))

        # Sécurité pour les bords de l'image
        trace_idx = max(0, min(trace_idx, num_traces - 1))
        sample_idx = max(0, min(sample_idx, num_samples - 1))

        # 3. Récupération de l'amplitude à ces indices
        amplitude = data[sample_idx, trace_idx]

        # 4. Mise à jour du label avec la nouvelle information d'amplitude (A)
        coord_text = f"X: {x_coord:.2f} | Y: {y_coord:.2f} | A: {amplitude}"
        self.coord_label.setText(coord_text)
        
        # La mise à jour de l'A-Scan (graphique de droite) reste fonctionnelle
        trace_data = data[:, trace_idx]
        y_values = np.linspace(y_axis_top, y_axis_bottom, num=num_samples)
        
        self.trace_plot.plot_trace(
                    trace_data, 
                    y_values, 
                    xlabel="Amplitude", 
                    ylabel=ylabel, 
                    y_cursor_pos=y_coord,
                    x_cursor_pos=amplitude  # <-- On ajoute la position en amplitude
                )

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

    def on_auto_t0_clicked(self):
        """
        Lance la détection automatique de T0 et met à jour l'interface.
        """
        if self.basalt.data is None:
            print("Veuillez d'abord charger un fichier.")
            return

        # 1. On appelle la méthode de détection de la classe Basalt
        detected_t0 = self.basalt.detect_t0()
        
        # 2. On met à jour le champ de texte Y0 avec la valeur trouvée
        self.line_edit_y0.setText(str(detected_t0))
        
        # 3. On appelle la fonction de mise à jour existante de Y0
        # pour que la nouvelle valeur soit prise en compte dans le traitement et l'affichage.
        # C'est une manière propre de déclencher la mise à jour complète.
        self.on_y0_edited()

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
        self.combo_sub_mean_mode.setEnabled(is_checked)
        
        # On appelle la logique de visibilité
        self.on_sub_mean_mode_changed(0) # 0 pour forcer la mise à jour de la visibilité
        
        self.update_display()

    def on_sub_mean_mode_changed(self, index):
        mode_text = self.combo_sub_mean_mode.currentText()
        self.basalt.traitementValues.sub_mean_mode = mode_text
        
        is_globale_mode = (mode_text == 'Globale')
        
        # On active les bons widgets et on cache les autres
        self.globale_options_widget.setVisible(is_globale_mode)
        self.globale_options_widget.setEnabled(self.checkbox_sub_mean.isChecked())
        
        self.mobile_options_widget.setVisible(not is_globale_mode)
        self.mobile_options_widget.setEnabled(self.checkbox_sub_mean.isChecked())
        
        if self.basalt.traitementValues.is_sub_mean:
            self.update_display()
        
    def on_trim_percent_edited(self):
        valeur = self._parse_input_to_float(self.line_edit_trim_percent.text(), default_on_error=0.0)
        self.basalt.traitementValues.sub_mean_trim_percent = valeur

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

    def on_migration_clicked(self):
        """Lance le processus de migration."""
        if self.basalt.data is None:
            print("Veuillez charger un fichier avant de lancer la migration.")
            return
            
        print("Lancement de la migration depuis l'interface...")
        # On délègue le travail à une nouvelle méthode dans Basalt
        self.basalt.run_migration()
        
        # Une fois la migration terminée, on met à jour l'affichage
        print("Mise à jour de l'affichage avec les données migrées...")
        self.redraw_plot() 

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

    def on_direction_mode_changed(self):
        """Met à jour le mode de direction du profil et rafraîchit l'affichage."""
        new_mode = 'normal' # Valeur par défaut
        if self.radio_direction_mirror_all.isChecked():
            new_mode = 'mirror_all'
        elif self.radio_direction_serpentine.isChecked():
            new_mode = 'mirror_serpentine'
            
        # Si le mode a réellement changé, on met à jour et on rafraîchit
        if self.basalt.traitementValues.profile_direction_mode != new_mode:
            self.basalt.traitementValues.profile_direction_mode = new_mode
            print(f"Mode de direction du profil changé en : {new_mode}")
            self.update_display()
            
    def on_interpolation_changed(self, mode: str):
        """Met à jour le mode d'interpolation et rafraîchit le graphique."""
        if self.basalt.data is None: return
        
        print(f"Mode d'interpolation changé en : '{mode}'")
        self.basalt.traitementValues.interpolation_mode = mode
        
        # C'est un changement léger, on appelle redraw_plot
        self.redraw_plot()

    def open_folder(self):
        """
        Ouvre une boîte de dialogue pour choisir un dossier, en commençant
        par le dernier dossier utilisé lors de la session précédente.
        """
        # 1. On détermine le chemin de départ pour la boîte de dialogue.
        # On essaie d'abord de prendre le dernier dossier mémorisé.
        start_path = self.basalt.last_used_folder 
        
        # 2. Si aucun dossier n'a été mémorisé (au tout premier lancement),
        # on utilise le dossier personnel de l'utilisateur comme alternative.
        if not start_path or not os.path.isdir(start_path):
            start_path = str(Path.home())

        # 3. On ouvre la boîte de dialogue en lui passant ce chemin de départ.
        selected_folder = QFileDialog.getExistingDirectory(self.window, 
                                                        "Sélectionner le dossier qui contient les traces GPR",
                                                        start_path) 
        
        # Le reste de la fonction est inchangé
        if selected_folder:
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
        self.last_used_folder: str = "" 
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

        self.current_file_index: int = 0

    def setFolder(self,folder:str):
        if os.path.isdir(folder):
            self.folder = folder
            self.last_used_folder = folder
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
        
    def getFilesFiltered(self, extension_filter: str, freq_key: str = None):
        """
        Récupère les fichiers du dossier, les filtre par extension et par clef de fréquence,
        puis les trie numériquement.
        
        Args:
            extension_filter (str): Le filtre d'extension (ex: "*.dzt").
            freq_key (str, optional): La clef de fréquence (ex: "_1"). Defaults to None.
        
        Returns:
            list: La liste finale des fichiers, triée.
        """
        # 1. On récupère la liste de fichiers bruts
        files_in_folder = self.getFilesInFolder(extension_filter)
        
        if not files_in_folder:
            return []

        # 2. On la filtre par clef de fréquence (si une clef est fournie)
        filtered_list = files_in_folder
        if freq_key is not None:
            filtered_list = [f for f in files_in_folder if f.stem.endswith(freq_key)]
        
        if not filtered_list:
            return []
            
        # 3. On trie la liste filtrée numériquement
        filtered_list.sort(key=lambda f: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f.stem)])
        
        return filtered_list

    
    def setSelectedFile(self, GPR_File:Path, radar : Radar, index_in_list: int = 0):

        self.antenna = radar
        self.selectedFile = GPR_File
        self.current_file_index = index_in_list
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
        
                # --- LOGIQUE D'INVERSION MIROIR ---
        mode = self.traitementValues.profile_direction_mode
        should_mirror = False
        
        if mode == 'mirror_all':
            should_mirror = True
        elif mode == 'mirror_serpentine':
            # On inverse les traces paires (index 1, 3, 5...)
            # La 1ère trace est à l'index 0 (impaire), la 2ème à 1 (paire), etc.
            if self.current_file_index % 2 == 1:
                should_mirror = True
                
        if should_mirror:
            self.traitement.apply_mirror()
        # --- FIN DE LA LOGIQUE ---

        if self.traitementValues.is_dewow_active : 
            self.traitement.dewow_filter()

        if self.traitementValues.is_sub_mean:
            self.traitement.sub_mean(
                mode=self.traitementValues.sub_mean_mode,
                window_size=self.traitementValues.sub_mean_window,
                trim_percent=self.traitementValues.sub_mean_trim_percent 
            )

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

    def detect_t0(self):
        """
        Détecte l'échantillon du pic de la première arrivée avec une zone de recherche contrôlée.
        """
        if self.data is None:
            return 0

        raw_data = self.data.dataFile
        t0_offset = 0

        if self.antenna == Radar.GSSI_FLEX:
            print("Cas GSSI Flex détecté. Analyse du canal sélectionné.")
            data_to_analyze = self._getFlexData(raw_data)
            if not self.boolFlex:
                t0_offset = raw_data.shape[0] // 2
        else:
            data_to_analyze = raw_data.copy()

        if data_to_analyze.size == 0:
            return 0

        # --- DÉBUT DE LA NOUVELLE LOGIQUE DE RECHERCHE ---

        # 1. Définir les limites de la recherche
        start_sample = 10  # On ignore toujours le tout premier échantillon
        
        # Par défaut, on cherche jusqu'à la fin du tableau analysé
        end_sample = data_to_analyze.shape[0] 
        
        # Règle spécifique pour les GSSI Flex : on ne dépasse pas l'échantillon 1022
        if self.antenna == Radar.GSSI_FLEX:
            end_sample = 1022
            print(f"Zone de recherche pour Flex limitée à [{start_sample}, {end_sample}]")

        # On s'assure que la zone de recherche est valide
        if start_sample >= end_sample:
            print("Avertissement : Zone de recherche invalide. Retour à 0.")
            return 0

        # 2. Le calcul se fait sur le tableau complet, mais la recherche est limitée
        abs_data = np.abs(data_to_analyze.astype(np.float32))
        mean_energy_trace = np.mean(abs_data, axis=1)
        
        # 3. On applique argmax UNIQUEMENT sur la tranche qui nous intéresse
        # L'indice retourné sera RELATIF au début de cette tranche (c'est-à-dire à start_sample)
        relative_index = np.argmax(mean_energy_trace[start_sample:end_sample])
        
        # 4. On calcule l'index local en ajoutant le point de départ de la recherche
        local_t0_sample = relative_index + start_sample
        
        # --- FIN DE LA NOUVELLE LOGIQUE ---

        # 5. On calcule l'indice absolu en ajoutant l'offset du canal (pour Flex)
        absolute_t0_sample = local_t0_sample + t0_offset
        
        print(f"T0 automatiquement détecté à l'échantillon absolu : {absolute_t0_sample}")
        return absolute_t0_sample

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

    def run_migration(self):
        """Prépare les paramètres et lance la migration de Kirchhoff."""
 
        if self.traitement is None or self.data is None: return

        header = self.data.header
        v = cste_global["c_lum"] / sqrt(self.traitementValues.epsilon)
        
        # --- Logique pour dt_s et dx_m ---
        time_window_ns = header.value_time
        total_samples_in_file = header.value_sample
        
        if self.antenna == Radar.GSSI_FLEX:
            time_window_ns /= 2.0
            total_samples_in_file /= 2.0
            
        dt_s = (time_window_ns * 1e-9) / total_samples_in_file if total_samples_in_file else 0
        dx_m = header.value_dist_total / header.value_trace if header.value_trace else 0
        # ---------------------------------
        
        aperture_m = 2.0 

        self.traitement.apply_kirchhoff_migration(dx=dx_m, dt=dt_s, v=v, aperture_m=aperture_m)

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

    def sub_mean(self, mode: str, window_size: int, trim_percent: float):
        """
        Soustrait la trace de fond en mode 'Globale' (moyenne tronquée) ou 'Mobile'.
        """
        print(f"Application du retrait de trace en mode '{mode}'...")
        data_float = self.data.astype(np.float32)

        if mode == 'Globale':
                proportion_to_cut = trim_percent / 100.0
                
                # --- LOGIQUE CONDITIONNELLE ---
                if proportion_to_cut > 0:
                    print(f"Calcul de la moyenne tronquée à {trim_percent}%...")
                    nt, nx = data_float.shape
                    mean_trace = np.zeros(nt, dtype=np.float32)
                    for i in range(nt):
                        mean_trace[i] = trim_mean(data_float[i, :], proportion_to_cut)
                else:
                    print("Calcul de la moyenne standard (pas d'élagage)...")
                    mean_trace = np.mean(data_float, axis=1) # Pas besoin de keepdims ici
                
                self.data = (data_float - mean_trace[:, np.newaxis]).astype(self.data.dtype)

                
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

    def apply_mirror(self):
        """
        Inverse l'ordre des traces sur l'axe horizontal (effet miroir).
        """
        print("Application de l'effet miroir sur les traces.")
        
        # Le slicing [:, ::-1] signifie :
        # : -> prend toutes les lignes (tous les samples)
        # ::-1 -> prend toutes les colonnes (les traces) mais avec un pas de -1,
        # ce qui inverse leur ordre.
        self.data = self.data[:, ::-1]
    
    def apply_kirchhoff_migration(self, dx: float, dt: float, v: float, aperture_m: float):
        """
        Applique la migration de Kirchhoff aux données.
        """
        print("Début de la migration de Kirchhoff...")
        nt, nx = self.data.shape

        if nt == 0 or dx == 0 or dt == 0 or v == 0:
            print("Erreur: Paramètres de migration invalides.")
            return
            
        aperture_traces = int(aperture_m / dx)
        migrated_data = np.zeros_like(self.data, dtype=np.float64)
        data_float = self.data.astype(np.float64)

        print(f"AVANT migration: min={np.min(data_float):.2f}, max={np.max(data_float):.2f}")

        # Appel à la fonction Numba, en passant self.y_offset
        migrated_data = _kirchhoff_migration_loop(
            data_float, migrated_data, nt, nx, dt, dx, v, aperture_traces, self.y_offset
        )
        
        print(f"APRÈS migration: min={np.min(migrated_data):.2f}, max={np.max(migrated_data):.2f}")
        self.data = migrated_data.astype(self.data.dtype)
        print("Migration terminée.")

class Graphique():
    def __init__(self, ax : plt.Axes, fig : Figure):
        self.vmin = -5e9
        self.vmax = 5e9
        self.contraste = 0.5
        self.fig : Figure = fig
        self.ax : plt.Axes = ax
        self.im = None

    def setVerticalgrid(self,flag : bool):
        self.ax.grid(visible=flag, axis='x',linewidth = 0.5, color = "black", linestyle ='-.')

    def setHorizontalgrid(self,flag : bool):
        self.ax.grid(visible=flag, axis='y',linewidth = 0.5, color = "black", linestyle ='-.')

    def plot(self, data, title:str, x_ticks: int, y_ticks: int, extent: list, xlabel: str, ylabel: str, show_x_ticks: bool, show_y_ticks: bool, interpolation_mode: str):
        titre_complet = "Scan : " + title
        
        if self.im is None:
            self.im = self.ax.imshow(data, cmap="gray", aspect="auto", interpolation=interpolation_mode, extent=extent)
            self.fig.suptitle(titre_complet, y=0.05, fontsize=12)
            self.ax.xaxis.set_label_position('top')
            self.ax.xaxis.set_ticks_position('top')
        else:
            self.im.set_data(data)
            self.im.set_extent(extent)
            self.fig.suptitle(titre_complet, y=0.05, fontsize=12)
            self.im.set_interpolation(interpolation_mode)

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

        try: #Optimisation de l'espace
            self.fig.tight_layout(rect=[0, 0.05, 1, 1])
        except Exception as e:
            print(f"Avertissement : tight_layout a échoué. {e}")

        self.ax.figure.canvas.draw()

    def plot_trace(self, trace_data, y_values, xlabel: str, ylabel: str,y_cursor_pos: float = None, x_cursor_pos: float = None):
        """
        Efface et redessine le graphique pour afficher une seule trace (courbe 1D).
        """
        # 1. On efface complètement les anciens tracés pour éviter les superpositions
        self.ax.cla()

        # 2. On trace les données : amplitude en X, profondeur/temps/sample en Y
        self.ax.plot(trace_data, y_values, color='blue', linewidth=0.8)
        
          # Dessine la ligne horizontale du curseur Y si une position est fournie
        if y_cursor_pos is not None:
            self.ax.axhline(y=y_cursor_pos, color='limegreen', linestyle=':', linewidth=1.5)
            
        # Dessine la ligne verticale du curseur X (amplitude) si une position est fournie
        if x_cursor_pos is not None:
            self.ax.axvline(x=x_cursor_pos, color='limegreen', linestyle=':', linewidth=1.5)


        # 3. On configure les axes
        self.ax.set_xlabel(xlabel)
        self.ax.xaxis.set_label_position('top')
        self.ax.xaxis.set_ticks_position('top')
        
        # 4. On inverse l'axe Y pour qu'il corresponde au radargramme (0 en haut)
        self.ax.invert_yaxis()
        
        vmin, vmax = self.getRangePlot()
        #self.ax.set_xlim(vmin, vmax)

        # On active une grille pour mieux lire les valeurs
        self.ax.grid(True, linestyle='--', linewidth=0.5)
        
        try: #Optimisation de l'espace
            self.fig.tight_layout(rect=[0, 0.05, 1, 1])
        except Exception as e:
            print(f"Avertissement : tight_layout a échoué. {e}")
            
        self.fig.canvas.draw()


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
