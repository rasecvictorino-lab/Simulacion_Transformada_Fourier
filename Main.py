import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. CONFIGURACION DE LA SIMULACION
# ==========================================================

Fs = 1000  # Frecuencia de muestreo (Hz)
T = 1 / Fs  # Periodo de muestreo (s)
N = 1000  # Numero de muestras
t = np.arange(0, N * T, T)  # Vector de tiempo
f = np.fft.fftshift(np.fft.fftfreq(N, d=T))  # Vector de frecuencias

# ==========================================================
# 2. DEFINICION DE LAS SEÑALES EN EL DOMINIO DEL TIEMPO
# ==========================================================

# Pulsos rectangulare de amplitud 1 y duracion 0.1s
pulso1 = (np.abs(t) <= 0.1).astype(float)

# Funcion de escalon unitario: vale 0 antes de t=0 y 1 desde t=0
escalon = (t >= 0).astype(float)

# Señal senoidal de 10 Hz
f0 = 10  # Frecuencia de la señal senoidal (Hz)
senoidal = np.sin(2 * np.pi * f0 * t)

señales = [
    ('Pulso rectangular', pulso1),
    ('Escalón unitario', escalon),
    ('Señal senoidal 10 Hz', senoidal)
]

# ==========================================================
# 3. TRANSFORMADA DE FOURIER MEDIANTE FFT
# ==========================================================

def calcular_fft(x):
    # Se centra el espectro para visualizar frecuencias negativas y positivas
    X = np.fft.fftshift(np.fft.fft(x)) / Fs
    magnitud = np.abs(X)
    fase = np.angle(X)
    
    return X, magnitud, fase


# ==========================================================
# 4. GRAFICAS EN TIEMPO Y FRECUENCIA
# ==========================================================

fig, axs = plt.subplots(3,3, figsize=(15, 11))

for i, (nombre, x) in enumerate(señales):
    X, magnitud, fase = calcular_fft(x)
    fase_grafica = np.where(magnitud > np.max(magnitud) * 1e-6, fase, np.nan)

    # Graficar señal en el tiempo
    axs[i, 0].plot(t, x)
    axs[i, 0].set_title(nombre + " - Tiempo")
    axs[i, 0].set_xlabel("Tiempo (s)")
    axs[i, 0].set_ylabel("Amplitud")
    axs[i, 0].grid(True)

    # Grafica de la magnitud del espectro
    axs[i, 1].plot(f, magnitud)
    axs[i, 1].set_title(nombre + " - Magnitud")
    axs[i, 1].set_xlabel("Frecuencia (Hz)")
    axs[i, 1].set_ylabel("Magnitud")
    axs[i, 1].set_xlim(-50, 50)
    axs[i, 1].grid(True)

# Grafica de fase; se ocultan valores donde la magnitud
# es muy pequeña para evitar interpretar fases inestables.

    axs[i, 2].plot(f, fase_grafica)
    axs[i, 2].set_title(nombre + " - Fase")
    axs[i, 2].set_xlabel("Frecuencia (Hz)")
    axs[i, 2].set_ylabel("Fase (radianes)")
    axs[i, 2].set_xlim(-50, 50)
    axs[i, 2].grid(True)

plt.tight_layout()
plt.show()

# ==========================================================
# 5. PROPIEDAD DE LINEALIDAD
# ==========================================================

# F{ax1 + b*x2} = a*F{x1} + b*F{x2}

x1 = np.sin(2 * np.pi * 10 * t)  # Señal 1: Senoidal de 10 Hz
x2 = np.sin(2 * np.pi * 20 * t)  # Señal 2: Senoidal de 20 Hz
a=2  # Escalar para x1
b=3  # Escalar para x2

fft_combinada = np.fft.fft(a * x1 + b * x2)
fft_separada = (
    a * np.fft.fft(x1) + b * np.fft.fft(x2)
)

error_linealidad = np.max(np.abs(fft_combinada - fft_separada)
)

print("\n--- Propiedad de Linealidad ---")
print("Error maximo:", error_linealidad)
print("Se cumple si el error es cercano a cero.")

# ==========================================================
# 6. PROPIEDAD DE DESPLAZAMIENTO TEMPORAL
# ==========================================================

# Desplazar una señal en el tiempo modifica la fase
# de su espectro, pero no su magnitud.
desplazamiento_muestras = 100  # Desplazamiento en muestras
x_original = np.sin(2 * np.pi * 10 * t)  # Señal original: Senoidal de 10 Hz
x_desplazada = np.roll(x_original, desplazamiento_muestras)  # Señal desplazada

X_original = np.fft.fft(x_original)
X_desplazada = np.fft.fft(x_desplazada)

error_desplazamiento = np.max(np.abs(np.abs(X_original) 
- np.abs(X_desplazada))
)

print("\n--- Propiedad de Desplazamiento Temporal ---")
print("Error maximo entre magnitudes:", error_desplazamiento)
print("La magnitud debe mantenerse aproximadamente igual.")

# ==========================================================
# 7. ESCALAMIENTO EN FRECUENCIA
# ==========================================================

# Comparamos dos senoidales: una de 10 Hz y otra de 20 Hz.
# Duplicar la frecuencia en el tiempo desplaza el pico
# del espectro de 10 Hz a 20 Hz.

seno_10 = np.sin(2 * np.pi * 10 * t)  # Señal de 10 Hz
seno_20 = np.sin(2 * np.pi * 20 * t)  # Señal de 20 Hz

F10 = np.abs(np.fft.fft(seno_10))
F20 = np.abs(np.fft.fft(seno_20))

frecuencias_fft = np.fft.fftfreq(N, d=T)

# Solo se buscan los picos de frecuencia positiva.
positivos = frecuencias_fft >= 0
f_positivas = frecuencias_fft[positivos]

pico_10 = f_positivas[np.argmax(F10[positivos])]
pico_20 = f_positivas[np.argmax(F20[positivos])]

print("\n--- Escalamiento en Frecuencia ---")
print("Pico de 10 Hz:", pico_10, "Hz")
print("Pico de 20 Hz:", pico_20, "Hz")

plt.figure(figsize=(10, 5))
plt.plot(f_positivas, F10[positivos], label='Senoidal 10 Hz')
plt.plot(f_positivas, F20[positivos], label='Senoidal 20 Hz')
plt.xlim(0, 40)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Comparacion de Escalamiento en Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

