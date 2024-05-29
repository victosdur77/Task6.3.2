import numpy as np
import matplotlib.pyplot as plt
from gudhi import RipsComplex
from gudhi import AlphaComplex
from gudhi.representations import DiagramSelector
import gudhi as gd
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
import math
from scipy import sparse
import ripser

ps = np.load("Robots.npy")
ds = [pairwise_distances(X).flatten() for X in ps[:,:,:2]]
maxd = np.max(np.concatenate(ds))

# topological descriptors calculation
def calculaDiagramaPersistencia(puntos,dimension,complex="alpha"):
    if complex not in ["rips","alpha"]:
        raise ValueError("The selected complex must be rips or alpha")
    elif complex=="alpha":
        alpha_complex = AlphaComplex(points=puntos) # 0ption 1: Using alpha complex
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=maxd)
    else:
        rips_complex = RipsComplex(points=puntos,max_edge_length=maxd) # Option 2: Using Vietoris-Rips complex
        simplex_tree = rips_complex.create_simplex_tree()
    diagrama_persistente = simplex_tree.persistence()
    persistence = simplex_tree.persistence_intervals_in_dimension(dimension)
    return persistence

def limitaDiagrama(Diagrama,maximaFiltracion,remove=False):
    if remove is False:
        infinity_mask = np.isinf(Diagrama) #Option 1:  Change infinity by a fixed value
        Diagrama[infinity_mask] = maximaFiltracion 
    elif remove is True:
        Diagrama = DiagramSelector(use=True).fit_transform([Diagrama])[0] #Option 2: Remove infinity bars
    return Diagrama

def calculaEntropia(persistentBarcode):
    l=[]
    for i in persistentBarcode:
        l.append(i[1]-i[0])
    L = sum(l)
    p=l/L
    entropia=-np.sum(p*np.log(p))
    return round(entropia,4)

def relative_entropy(persistentBarcode):
    entropia=calculaEntropia(persistentBarcode) / len(persistentBarcode)
    return round(entropia,4)

def calculate_lowerStair_PDs(t,x):
    N = x.shape[0]
    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    dgms = ripser.ripser(D, maxdim=1, distance_matrix=True)['dgms']
    return dgms

def diagram_lowerstair_dimension(Diagramas,dimension):
    dgm=Diagramas[dimension]
    dgm = dgm[dgm[:, 1]-dgm[:, 0] > 1e-3, :]
    return dgm

# gráficos

def dibujaNubePuntosInstante(time,robotVision=None,vision_radius=5,field_of_view=np.pi/2,ids=False):
    instante = ps[time]
    x=instante[:,0]
    y=instante[:,1]
    angle=instante[:,2]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color='blue', label=f"Initial time: {time}")
    plt.quiver(x, y, np.cos(angle), np.sin(angle), color="green",
            angles='xy', scale_units='xy', scale=5, width=0.003, headwidth=3, headlength=3)
    if ids is True:
        for i in range(len(x)):
            plt.text(x[i], y[i]+0.1, str(i), fontsize=7, ha='center', va='bottom')
    
    if robotVision is not None: 
        xrobot=x[robotVision]
        yrobot=y[robotVision]
        orientation=angle[robotVision]
        arc_points = [[xrobot, yrobot]]  
        
        num_points = 50  
        for i in range(num_points + 1):
            angulos = orientation + field_of_view / 2 - (i / num_points) * field_of_view
            arc_points.append([xrobot + vision_radius * np.cos(angulos), yrobot + vision_radius * np.sin(angulos)])
        arc_points.append([xrobot, yrobot])  
        arc_points = np.array(arc_points)
        # plt.plot(arc_points[:, 0], arc_points[:, 1], 'b-', alpha=0.3) 
        plt.fill(arc_points[:, 0], arc_points[:, 1], color='blue', alpha=0.1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Point cloud in time: {time}')
    
def dibujaNubePuntos2Instantes(time1,time2):

    instante1 = ps[time1]
    instante2 = ps[time2]
    x1=instante1[:,0]
    y1=instante1[:,1]
    angle1=instante1[:,2]

    x2=instante2[:,0]
    y2=instante2[:,1]
    angle2=instante2[:,2]

    maxX=max(max(x1),max(x2)) + 1
    maxY=max(max(y1),max(y2)) + 1
    minX=min(min(x1),min(x2)) - 1
    minY=min(min(y1),min(y2)) - 1

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))  
    axs[0].scatter(x1, y1, color='blue', label=f"Initial time: {time1}")
    axs[0].quiver(x1, y1, np.cos(angle1), np.sin(angle1), color="red",
                  angles='xy', scale_units='xy', scale=5, width=0.003, headwidth=3, headlength=3)
    axs[0].set_title(f'Initial time: {time1}')  
    axs[0].set_xlim(minX,maxX)
    axs[0].set_ylim(minY,maxY)
    for i in range(len(x1)):
        axs[0].text(x1[i], y1[i]+0.1, str(i), fontsize=7, ha='center', va='bottom')

    axs[1].scatter(x2, y2, color='blue', label=f"End time: {time2}")
    axs[1].quiver(x2, y2, np.cos(angle2), np.sin(angle2), color="red",
                  angles='xy', scale_units='xy', scale=5, width=0.003, headwidth=3, headlength=3)
    axs[1].set_title(f'End time: {time2}') 
    axs[1].set_xlim(minX,maxX)
    axs[1].set_ylim(minY,maxY)
    for i in range(len(x2)):
        axs[1].text(x2[i], y2[i]+0.1, str(i), fontsize=7, ha='center', va='bottom')

    axs[2].quiver(x1, y1, np.cos(angle1), np.sin(angle1), color="blue",
                  angles='xy', scale_units='xy', scale=5, width=0.003, headwidth=3, headlength=3, label=f'Initial time: {time1}')
    axs[2].quiver(x2, y2, np.cos(angle2), np.sin(angle2), color="red",
                  angles='xy', scale_units='xy', scale=5, width=0.003, headwidth=3, headlength=3, label=f'End time: {time2}')
    for i in range(len(x1)):
         axs[2].plot([x1[i], x2[i]], [y1[i], y2[i]], color='gray', linestyle='--',linewidth=0.5,alpha=0.5)
    axs[2].set_xlim(minX,maxX)
    axs[2].set_ylim(minY,maxY)
    axs[2].legend()
    axs[2].set_title(f'Movements betweent time {time1} and {time2}') 
    plt.tight_layout()
    plt.show()

def dibujaPersisteceDiagram(time):
    nube_puntos=ps[time,:,:2]
    persistence = calculaDiagramaPersistencia(nube_puntos,0)
    gd.plot_persistence_diagram(persistence)
    plt.title(f"Persistent diagram for time {time}")

def dibujaPersisteceBarcode(time):
    nube_puntos=ps[time,:,:2]
    persistence = calculaDiagramaPersistencia(nube_puntos,0)
    persistenciaL=limitaDiagrama(persistence,maxd)
    entropia=calculaEntropia(persistenciaL)
    gd.plot_persistence_barcode(persistenciaL)
    plt.title(f"Persistent barcode for time {time}; Entropy: {entropia}")
    

def dibujaEntropyTimeSerie(entropy):
    plt.plot(entropy,marker='o')
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.title(f'Topological entropy time series of persistent diagram')
    plt.grid(True)
    plt.show()

def dibujaEntropyTimeSerieInteractive(entropy):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(0,100), 
            y=entropy,
            mode='lines+markers',
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        xaxis_title='Time',
        yaxis_title='Entropy',
        title=f'Topological entropy time series of persistent diagram'
    )
    fig.show()

#robots in field of vision
def calculate_robots_in_field_vision(time, robot,vision_radius=5,field_of_view=np.pi/2,printing=False):
    robots_en_campo = []
    instante = ps[time]
    x=instante[:,0]
    y=instante[:,1]
    angle=instante[:,2]
    xObjetivo = x[robot]
    yObjetivo = y[robot]
    angleObjetivo = angle[robot]
    angulo_inicio = angleObjetivo - field_of_view / 2
    angulo_fin = angleObjetivo + field_of_view / 2
    for i in range(len(x)):
        if i == robot:
            continue
        
        robot_x, robot_y = x[i], y[i]
        distancia = calculate_distance(xObjetivo,yObjetivo,robot_x,robot_y)
        if distancia > vision_radius:
            continue
        
        angulo_robot = np.arctan2(robot_y - yObjetivo, robot_x - xObjetivo)
        angulo_relativo = (angulo_robot - angleObjetivo + 2 * np.pi) % (2 * np.pi)
        angulo_inicio_relativo = (angulo_inicio - angleObjetivo + 2 * np.pi) % (2 * np.pi)
        angulo_fin_relativo = (angulo_fin - angleObjetivo + 2 * np.pi) % (2 * np.pi)
        if angulo_inicio_relativo < angulo_fin_relativo:
            if angulo_inicio_relativo <= angulo_relativo <= angulo_fin_relativo:
                robots_en_campo.append(i)
        else:  # Case when field vision cross 0 radians
            if angulo_relativo >= angulo_inicio_relativo or angulo_relativo <= angulo_fin_relativo:
                robots_en_campo.append(i)
    if printing is True:
        print(f"Time {time}. Robots in the robot's {robot} field of vision:", robots_en_campo)
    return robots_en_campo


# distancias y angulos
def calculate_distance(x1, y1, x2, y2):
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distancia

def calculate_angle(x, y, orientation, x2, y2):
    angle_to_point = np.arctan2(y2 - y, x2 - x)
    relative_angle = angle_to_point - orientation
    return relative_angle

def transform_angle(angulo):
    while angulo < 0:
        angulo += 360
    if angulo <= 180:
        anguloFinal = angulo
    else:
        anguloFinal = 360 - angulo
    return anguloFinal