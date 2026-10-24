"""Laboratorio 1 -- Python y NumPy para el trabajo numérico: la taza de café que se enfría.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================================
# Constantes y funciones de base -- no modificar.
# 
# IMPORTANTE: lea bien qué significa cada constante y qué realiza cada 
# función, ya que estos serán usados a lo largo del laboratorio.
# ==========================================================================

# Temperatura de la habitación, en grados Celsius
T_AMB = 22.0

# Temperatura del café en el instante en que se sirvió, en t = 0 minutos.
T0 = 92.0

# Constante de de tasa de enfriamiento
K_REAL = 0.030

# Temperatura objetivo, a la que el café se vuelve bebible, en grados Celsius.
OBJETIVO = 65.0

# Tiempos en los que el termómetro anotó una temperatura, en minutos.
TIEMPOS_REG = [
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
    22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0,
]

# Las 21 temperaturas registradas, en grados Celsius, una por cada entrada de
# TIEMPOS_REG.
TEMPS_REG = [
    91.7, 87.8, 83.8, 80.3, 76.7, 73.6, 71.0, 68.0, 65.3, 65.9, 60.3,
    58.2, 56.2, 53.9, 52.1, 50.5, 48.9, 47.8, 46.2, 44.5, 43.0,
]


def modelo(t, T0=T0, T_amb=T_AMB, k=K_REAL):
    """Ley de enfriamiento de Newton: la temperatura predicha en el tiempo ``t``.
    """
    return T_amb + (T0 - T_amb) * np.exp(-k * t)


def linealizar(temps, T_amb=T_AMB):
    """Endereza la curva de enfriamiento tomando un logaritmo.

    El modelo es ``T - T_amb = (T0 - T_amb) * exp(-k t)``.  Tomar el logaritmo
    natural de ambos lados convierte el producto en suma y la exponencial en una
    simple multiplicación:

        ln(T - T_amb) = ln(T0 - T_amb) - k * t

    El lado izquierdo es lo que calcula esta función.  Lee el lado derecho como
    ``z = intercepto + pendiente * t``: una recta cuya pendiente es ``-k``.  Esa
    es toda la idea de la Parte E; ajustar una recta es fácil, así que ajustamos
    la recta y leemos la tasa de enfriamiento en su pendiente.

    Devuelve un arreglo de NumPy de la misma longitud que ``temps``.
    """
    return np.log(np.asarray(temps, dtype=float) - T_amb)


# --------------------------------------------------------------------------
# Problema 1 -- el bucle `while`
# --------------------------------------------------------------------------
def minutos_hasta_bebible(T0=T0, T_amb=T_AMB, k=K_REAL, objetivo=OBJETIVO):
    """¿Cuántos minutos enteros hasta que el café modelado llega a ``objetivo``?

    Parámetros
    ----------
    T0 : float
        Temperatura en el tiempo cero, en grados Celsius.
    T_amb : float
        Temperatura de la habitación, en grados Celsius.
    k : float
        Constante de la tasa de enfriamiento, en 1/minuto.
    objetivo : float
        La temperatura que estamos esperando, en grados Celsius.

    Devuelve
    --------
    int
        El menor número entero de minutos ``n`` para el cual
        ``modelo(n) <= objetivo``.

    Notas
    -----
    * Un bucle ``while`` repite su cuerpo mientras una condición siga siendo
      verdadera::

          while condicion:
              cuerpo

      La condición se prueba *antes* de cada pasada, así que un bucle cuya
      condición es falsa al principio se ejecuta cero veces.

    * Use aquí un ``while`` y no un ``for``, porque no se sabe de antemano cuántos
      minutos tardará: esa cuenta es la respuesta. Esa es la idea principal de usos 
      de estos tipos de bucles: 
      ``for`` cuando el número de repeticiones se conoce por adelantado,
      ``while`` cuando no.

    * Todo bucle ``while`` necesita algo en su cuerpo que a la larga vuelva
      falsa la condición.  Aquí eso es ``minutos += 1``. Si se olvida, el bucle no
      termina nunca; este es el error más común con ``while``.

    * Empieza en ``minutos = 0`` y avanza de un minuto en un minuto.  La
      respuesta es un número entero de minutos, así que devuelve un ``int``, no
      un ``float``.
    """
    # Problema 1: cuenta minutos con un bucle while hasta que modelo(...) <= objetivo.
    minutos = 0
    while modelo(minutos, T0, T_amb, k) > objetivo:
        minutos += 1
    return minutos


# --------------------------------------------------------------------------
# Problema 2 -- el bucle `for x in ...`
# --------------------------------------------------------------------------
def primera_por_debajo(temps, objetivo=OBJETIVO):
    """Índice de la primera temperatura registrada igual o menor que ``objetivo``.

    Parámetros
    ----------
    temps : list of float
        Las temperaturas registradas, en orden.
    objetivo : float
        El umbral, en grados Celsius.

    Devuelve
    --------
    int
        La posición de la primera entrada con ``temps[i] <= objetivo``, o ``-1``
        si no hay ninguna.

    Notas
    -----
    * ``for x in secuencia:`` ejecuta su cuerpo una vez por cada elemento de la
      secuencia, ligando el elemento a ``x``.  Funciona igual con listas, tuplas,
      cadenas y arreglos de NumPy, y a diferencia de ``while`` se detiene solo
      cuando la secuencia se acaba.

    * Necesitas la *posición*, no el valor, así que combina el bucle con
      ``enumerate``::

          for i, valor in enumerate(temps):
              ...

      ``enumerate`` te entrega el índice y el elemento juntos.  Escribir
      ``for i in range(len(temps))`` e indexar a mano también funciona, pero
      ``enumerate`` es lo que uno lee en el código de otras personas.

    * Un ``return`` dentro del bucle sale de toda la función de inmediato, y así
      es como la función se detiene en la *primera* coincidencia y no en la última.

    * Una búsqueda que no encuentra nada tiene que devolver algo de todos modos.
      La convención que se usa aquí es ``-1``.  Devolver ``None`` en su lugar
      haría que quien llama fallara más tarde con un mensaje confuso, en vez de
      aquí con uno claro.
    """
    # Problema 2: recorre las lecturas con un bucle for y devuelve el primer índice.
    for i, valor in enumerate(temps):
        if valor <= objetivo:
            return i
    return -1


# --------------------------------------------------------------------------
# Problema 3 -- list.append
# --------------------------------------------------------------------------
def simular_con_append(n_minutos=40, T0=T0, T_amb=T_AMB, k=K_REAL):
    """La temperatura modelada cada minuto entero, construida valor a valor.

    Parámetros
    ----------
    n_minutos : int
        El último minuto que hay que simular.
    T0, T_amb, k : float
        Los parámetros del modelo, como en :func:`modelo`.

    Devuelve
    --------
    list of float
        Una lista de Python de longitud ``n_minutos + 1`` con la temperatura
        modelada en los minutos ``0, 1, ..., n_minutos``.  Ambos extremos están
        incluidos, y por eso la longitud es ``n + 1`` y no ``n``.

    Notas
    -----
    * ``lst.append(v)`` añade ``v`` al final de la lista ``lst`` y la deja una
      entrada más larga.  Modifica la lista *en el sitio*.

    * ``append`` devuelve ``None``.  Por eso::

          valores = valores.append(x)     # MAL -- valores ahora es None

      es un desastre silencioso: la lista se tira y se reemplaza por ``None``, y
      el error solo aparece varias líneas después.  Llámalo como una instrucción
      sola en su propia línea y nunca asignes su resultado.

    * El patrón es siempre el mismo: crea una lista vacía ``valores = []`` antes
      del bucle, añade dentro del bucle y devuelve la lista después.

    * Hacer crecer una lista así está bien para unos pocos miles de entradas.
    """
    # Problema 3: construye la lista con un bucle y .append
    valores = []
    for minuto in range(n_minutos + 1):
        valores.append(float(modelo(minuto, T0, T_amb, k)))
    return valores


# --------------------------------------------------------------------------
# Problema 4 -- la comprensión de listas
# --------------------------------------------------------------------------
def simular_por_comprension(n_minutos=40, T0=T0, T_amb=T_AMB, k=K_REAL):
    """Exactamente el resultado de :func:`simular_con_append`, en una expresión.

    Los parámetros y el valor devuelto son idénticos a los de
    :func:`simular_con_append`.

    Notas
    -----
    * Una *comprensión de listas* construye una lista a partir de un bucle en
      una sola expresión::

          [expresion for elemento in secuencia]

      Se lee de izquierda a derecha como "la ``expresion``, para cada
      ``elemento`` de la ``secuencia``".  El patrón de tres líneas
      crear-recorrer-añadir de la Tarea 3 se reduce a una línea.

    * Un filtro opcional va al final::

          [T for T in temps if T > 65.0]

      conserva solo las entradas que pasan la prueba.  Sin ``if``, cada elemento
      aporta exactamente una entrada, así que el resultado tiene la misma
      longitud que la entrada.

    * Usa ``range(n_minutos + 1)`` como secuencia para obtener los minutos
      ``0, 1, ..., n_minutos``.  ``range`` se detiene *antes* de su argumento,
      así que el ``+ 1`` es lo que incluye el minuto final.  El mismo desfase de
      uno reaparece en la Tarea 8 con ``np.arange``.

    * Una comprensión es la herramienta correcta cuando cada entrada se calcula
      con independencia de las demás.  Es la herramienta equivocada cuando una
      entrada depende de la anterior, como en un método iterativo.  Para esos
      quédate con el bucle explícito; las Prácticas 3 a 9 están llenas de ellos.

    * Comprueba tú mismo que esto devuelve los mismos números que la Tarea 3.
      La batería de pruebas lo verifica, y tú también deberías.
    """
    # Problema 4: devuelve la misma lista de la Tarea 3, como una comprensión.
    return [float(modelo(minuto, T0, T_amb, k)) for minuto in range(n_minutos + 1)]


# ==========================================================================
# PARTE 2 -- Arreglos con Numpy
# ==========================================================================


# --------------------------------------------------------------------------
# Problema 5 -- np.array
# --------------------------------------------------------------------------
def registro_a_arreglo(temps):
    """Convierte las temperaturas registradas en un arreglo de NumPy de floats.

    Parámetros
    ----------
    temps : list of float
        Las lecturas del registrador.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(len(temps),)``, dtype ``float64``.

    Notas
    -----
    * ``np.array(obj, dtype=...)`` construye un arreglo *nuevo* a partir de
      ``obj``: una lista, una tupla, una lista anidada u otro arreglo.
      ``dtype`` dice qué tipo deben tener las entradas; pasa ``float`` para
      asegurarte de obtener ``float64`` aunque todos los valores de la lista
      parezcan enteros.

    * Dos atributos describen cualquier arreglo, y los leerás constantemente:

      - ``a.shape`` es una *tupla* de longitudes.  Un vector de 21 entradas
        tiene forma ``(21,)``; fíjate en la coma final, que es como Python
        escribe una tupla de un solo elemento.  Una matriz de 21 por 2 tiene
        forma ``(21, 2)``.
      - ``a.dtype`` es el tipo de cada entrada, aquí ``float64``.

    * ``np.array`` siempre **copia**.  El arreglo que devuelve es independiente
      de la lista de la que salió: cambiar uno no toca al otro.  Esa es la
      diferencia con ``np.asarray`` de la Tarea 6, y es todo el contenido de
      ambas tareas.

    * Usa ``dtype=float`` de forma explícita.  Un arreglo de ``int64`` parece
      inofensivo hasta que una división trunca en silencio y pierdes la mitad de
      tu exactitud.
    """
    # Problema 5: construye un arreglo float64 a partir de la lista.
    return np.array(temps, dtype=float)


# --------------------------------------------------------------------------
# Problema 6 -- np.asarray
# --------------------------------------------------------------------------
def como_arreglo_float(valores):
    """Acepta una lista *o* un arreglo y devuelve un arreglo de floats, barato.

    Parámetros
    ----------
    valores : array_like
        Una lista, una tupla o un arreglo de NumPy ya existente.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(len(valores),)``, dtype ``float64``.  Si ``valores`` ya es un
        arreglo ``float64``, se devuelve ese mismísimo objeto, sin copia.

    Notas
    -----
    * ``np.asarray(obj, dtype=float)`` convierte ``obj`` en arreglo **solo si no
      es ya uno del tipo correcto**.  Ante una lista hace el mismo trabajo que
      ``np.array``; ante un arreglo que ya encaja devuelve ese arreglo sin
      tocarlo.

    * Eso lo convierte en la primera línea estándar de una función numérica::

          x = np.asarray(x, dtype=float)

      La función funciona entonces tanto si quien llama pasó una lista como si
      pasó un arreglo, y no cuesta nada de más en el caso habitual de recibir un
      arreglo.  Todas las soluciones desde la Práctica 2 en adelante empiezan
      así, y por eso merece una tarea propia.

    * La trampa es la otra cara del ahorro: como no se hizo copia, el arreglo
      devuelto puede ser el arreglo de quien llama.  Escribir en él cambia los
      datos de esa persona.  Si piensas modificar los valores, usa ``np.array``
      (que siempre copia) o ``np.asarray(...).copy()``.

    * Dos objetos se comparan por identidad con ``a is b``, que pregunta "¿son
      el mismo objeto?", frente a ``a == b``, que compara valores.  Así es como
      la prueba de esta tarea distingue una función de la otra.
    """
    # Problema 6: convierte sin copiar cuando sea posible.
    return np.asarray(valores, dtype=float)


# --------------------------------------------------------------------------
# Problema 7 -- np.linspace
# --------------------------------------------------------------------------
def malla_fina_de_tiempo(t_fin=40.0, num=401):
    """Una malla densa de tiempos para dibujar la curva suave del modelo.

    Parámetros
    ----------
    t_fin : float
        El último tiempo de la malla, en minutos.
    num : int
        Cuántos puntos debe contener la malla.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(num,)``, que va de ``0.0`` a ``t_fin`` inclusive.

    Notas
    -----
    * ``np.linspace(inicio, fin, num)`` devuelve ``num`` valores igualmente
      espaciados de ``inicio`` a ``fin``, **con ambos extremos incluidos**.  Tú
      eliges el número de puntos y el espaciado se deduce:

          h = (fin - inicio) / (num - 1)

      con ``num - 1`` en el denominador, no ``num``: cubrir un intervalo con
      ``n`` subintervalos requiere ``n + 1`` puntos.  Equivocarse en uno ahí es
      como una regla de integración termina discretamente desviada un pequeño
      porcentaje, que es justo el tema de la Práctica 9.

    * El valor por omisión es ``num=50`` si lo dejas fuera, que casi nunca es lo
      que quieres.  Pásalo siempre.

    * Usa ``np.linspace`` siempre que quieras *cubrir un intervalo*, porque es el
      que cae exactamente en los dos extremos.  Usa ``np.arange`` (Tarea 8)
      cuando lo que te importa es el *espaciado*.

    * 401 puntos en 40 minutos son un espaciado de 0.1 min.  Una curva dibujada
      necesita puntos suficientes para que el ojo no vea los segmentos rectos
      entre ellos; unos pocos cientos es la respuesta habitual.
    """
    # Problema 7: construye la malla con np.linspace.
    return np.linspace(0.0, t_fin, num)


# --------------------------------------------------------------------------
# Problema 8 -- np.arange
# --------------------------------------------------------------------------
def malla_gruesa_de_tiempo(t_fin=40.0, paso=5.0):
    """Los tiempos 0, 5, 10, ... hasta ``t_fin`` inclusive.

    Parámetros
    ----------
    t_fin : float
        El último tiempo que debe aparecer en la malla, en minutos.
    paso : float
        La separación entre tiempos consecutivos, en minutos.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(t_fin / paso + 1,)``, con los valores de ``0.0`` a ``t_fin``.

    Notas
    -----
    * ``np.arange(inicio, fin, paso)`` devuelve los valores ``inicio``,
      ``inicio + paso``, ``inicio + 2*paso``, ... mientras se mantengan
      **estrictamente por debajo** de ``fin``.  Aquí fijas el espaciado y el
      número de puntos se deduce: lo contrario de ``np.linspace``.

    * **El extremo derecho queda excluido.**  ``np.arange(0.0, 40.0, 5.0)`` se
      detiene en 35.0.  Para incluir 40.0 tienes que pedir un fin más allá, y la
      manera habitual es ``np.arange(0.0, t_fin + paso, paso)``.  Lee eso con
      cuidado: es el error más común de todo NumPy, y es la razón de que esta
      tarea vaya junto a la Tarea 7.

    * Una segunda trampa: con un ``paso`` fraccionario, el redondeo decide si el
      último valor se cuela por debajo de ``fin`` o no, así que la longitud del
      resultado puede cambiar en uno por motivos que no tienen nada que ver con
      tu intención.  Que ``np.arange(0, 1, 0.1)`` dé 10 entradas y no 11 es
      suerte, no diseño.  Cuando la cuenta importa, usa ``np.linspace``.

    * ``np.arange`` es la contraparte para arreglos del ``range`` incorporado, y
      comparte su extremo superior excluyente: el mismo desfase de uno que ya
      viste en la Tarea 4.
    """
    # Problema 8: construye la malla con np.arange, incluyendo t_fin.
    return np.arange(0.0, t_fin + paso, paso)


# --------------------------------------------------------------------------
# Problema 9 -- np.zeros, np.ones, np.full
# --------------------------------------------------------------------------
def columna_constante(n, valor=0.0):
    """Un arreglo de longitud ``n`` en el que cada entrada vale ``valor``.

    Parámetros
    ----------
    n : int
        La longitud del arreglo.
    valor : float
        El valor con el que llenarlo.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(n,)``, dtype ``float64``, cada entrada igual a ``valor``.

    Notas
    -----
    * Tres constructores crean un arreglo de tamaño conocido con contenido
      conocido:

      - ``np.zeros(n)``       -- ``n`` entradas, todas ``0.0``
      - ``np.ones(n)``        -- ``n`` entradas, todas ``1.0``
      - ``np.full(n, valor)`` -- ``n`` entradas, todas ``valor``

      Los tres devuelven ``float64`` por omisión, salvo ``np.full``, que toma su
      dtype de ``valor``, así que ``np.full(3, 7)`` da enteros.  Pasa
      ``dtype=float`` ante la duda.

    * El primer argumento es una *forma*.  Dale un entero para un vector, o una
      tupla para una matriz: ``np.zeros((3, 4))`` es una matriz de ceros de 3
      por 4.  El paréntesis doble no es una errata: la tupla es un argumento.

    * Esto es *reserva previa*: crea todo el arreglo de una vez y luego llénalo.
      Es la contraparte para arreglos del patrón lista-vacía-y-append de la
      Tarea 3, y es lo que quieres siempre que el tamaño final se conozca de
      antemano.

    * ``np.ones`` en particular es como se construye una columna de unos para un
      ajuste por mínimos cuadrados, que es justo lo que necesita la Parte E,
      aunque la Tarea 12 llega allí por un camino más corto.

    * implemente esto con el que mejor encaje de los tres.  Basta una llamada.
    """
    # Problema 9: devuelve un arreglo constante de longitud n.
    return np.full(n, valor, dtype=float)


# --------------------------------------------------------------------------
# Problema 10 -- np.empty
# --------------------------------------------------------------------------
def columna_del_modelo(tiempos):
    """La temperatura modelada en cada entrada de ``tiempos``, escrita en un
    arreglo reservado de antemano con ``np.empty``.

    Parámetros
    ----------
    tiempos : array_like
        Los tiempos en los que evaluar el modelo, en minutos.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(len(tiempos),)``, dtype ``float64``, con la entrada ``i`` igual
        a ``modelo(tiempos[i])``.

    Notas
    -----
    * ``np.empty(n)`` reserva memoria para ``n`` floats y te la entrega **sin
      fijar los valores**.  Es la manera más rápida de obtener un arreglo de un
      tamaño dado, porque se salta el trabajo de escribir ceros en él.

    * Lo que contiene, por tanto, son los bytes que hubiera en esa dirección: no
      ceros, ni basura en un sentido pintoresco, sino números reales que dejó
      otra cosa, a veces incluidos ``nan`` o valores del orden de ``1e-310``.
      **Tienes que escribir en cada entrada antes de leer ninguna.**  Un arreglo
      de ``np.empty`` que solo llenas en parte es un error que produce
      respuestas equivocadas de aspecto verosímil, que es la peor clase.

    * Si no tienes la certeza de que llenarás todas las casillas, usa
      ``np.zeros``.  El costo es una pasada por la memoria y la ventaja es que
      una casilla olvidada aparece como un 0.0 evidente.

    * Llena aquí el arreglo con un bucle ``for`` explícito sobre los índices,
      para haber escrito una vez la versión lenta::

          salida = np.empty(len(tiempos))
          for i in range(len(tiempos)):
              salida[i] = ...

      Después fíjate en que ``modelo(np.asarray(tiempos))`` hace todo eso en una
      sola llamada, porque ``np.exp`` trabaja sobre arreglos enteros.  Eso es
      *vectorización*, y desde la Práctica 2 en adelante es como deberías
      escribir todo lo que no sea genuinamente iterativo.
    """
    # Problema 10: reserva con np.empty y después llena todas las entradas.
    tiempos = np.asarray(tiempos, dtype=float)
    salida = np.empty(tiempos.shape, dtype=float)
    for i in range(tiempos.size):
        salida[i] = modelo(tiempos[i])
    return salida


# --------------------------------------------------------------------------
# Problema 11 -- np.zeros_like
# --------------------------------------------------------------------------
def contenedor_de_residuos(temps):
    """Un arreglo de ceros con la misma forma y dtype que ``temps``.

    Parámetros
    ----------
    temps : array_like
        El arreglo cuya forma y dtype hay que copiar.

    Devuelve
    --------
    numpy.ndarray
        La misma forma y dtype que ``np.asarray(temps, dtype=float)``, con cada
        entrada igual a ``0.0``.

    Notas
    -----
    * ``np.zeros_like(a)`` es ``np.zeros(a.shape, dtype=a.dtype)`` escrito sin
      tener que mencionar ninguno de los dos.  Copia la *forma* y el *dtype* de
      ``a`` y llena el resultado de ceros; no copia los valores.

    * La idea es que dejes de repetirte.  Escribir ``np.zeros(len(temps))``
      funciona hoy y se rompe el día en que ``temps`` pase a ser bidimensional o
      complejo.  ``np.zeros_like(temps)`` lo sigue.  La Práctica 6 depende de
      esto cuando sus arreglos se vuelven complejos a mitad de camino.

    * ``np.ones_like`` (Tarea 12) y ``np.full_like(a, v)`` son los otros dos de
      la familia, y ``np.empty_like`` la completa con la misma advertencia de la
      Tarea 10 adjunta.

    * El dtype se hereda, trampas incluidas:
      ``np.zeros_like(np.array([1, 2]))`` es un arreglo de *enteros*, y asignarle
      0.5 guarda 0.  Convierte primero con ``np.asarray(temps, dtype=float)``,
      como exige el contrato de esta función.
    """
    # Problema 11: devuelve un arreglo de ceros con la forma de temps.
    temps = np.asarray(temps, dtype=float)
    return np.zeros_like(temps)


# --------------------------------------------------------------------------
# Problema 12 -- np.ones_like
# --------------------------------------------------------------------------
def columna_de_unos(tiempos):
    """Una columna de unos, una por cada entrada de ``tiempos``.

    Parámetros
    ----------
    tiempos : array_like
        Los tiempos de las mediciones.  Solo se usan su forma y su dtype.

    Devuelve
    --------
    numpy.ndarray
        La misma forma que ``tiempos``, dtype ``float64``, cada entrada ``1.0``.

    Notas
    -----
    * ``np.ones_like(a)`` es a ``np.ones`` lo que ``np.zeros_like`` es a
      ``np.zeros``: misma forma, mismo dtype, cada entrada ``1.0``.

    * Una columna de unos no es trabajo inútil.  Cuando ajustas la recta
      ``z = c0 + c1 * t``, el término constante ``c0`` multiplica a 1 en cada
      medición.  Escribir eso como una columna convierte el ajuste en una sola
      ecuación matricial, que es lo que arman las Tareas 19 a 21.  La columna de
      unos *es* el intercepto.

    * Constrúyela a partir de ``tiempos``, no de una longitud que escribiste a
      mano, por la razón dada en la Tarea 11: así el código sigue funcionando
      cuando los datos cambian de tamaño.

    * Convierte primero con ``dtype=float``.  Un registro de marcas de tiempo
      enteras te daría si no una columna de unos enteros, y la matriz de diseño
      de la Tarea 19 saldría como una matriz de enteros.
    """
    # Problema 12: devuelve un arreglo float de unos con la forma de tiempos.
    tiempos = np.asarray(tiempos, dtype=float)
    return np.ones_like(tiempos)


# ==========================================================================
# PARTE 3 -- Operaciones elemento a elemento y reducciones
# ==========================================================================


# --------------------------------------------------------------------------
# Problema 13 -- np.abs
# --------------------------------------------------------------------------
def residuos_absolutos(temps, temps_modelo):
    """Qué tan lejos está cada lectura del modelo, sin mirar la dirección.

    Parámetros
    ----------
    temps : array_like
        Las temperaturas registradas.
    temps_modelo : array_like
        Las temperaturas modeladas en los mismos tiempos.  Misma longitud que
        ``temps``.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(len(temps),)``, con la entrada ``i`` igual a
        ``abs(temps[i] - temps_modelo[i])``.

    Notas
    -----
    * La aritmética entre dos arreglos de la misma forma es *elemento a
      elemento*: ``a - b`` resta entrada por entrada y da un arreglo de esa
      misma forma.  De tu lado no hay ningún bucle.

    * ``np.abs(a)`` toma el valor absoluto de cada entrada y devuelve un arreglo
      de la misma forma.  Las funciones que se comportan así -- ``np.abs``,
      ``np.exp``, ``np.sin``, ``np.sqrt``, ``np.log`` -- se llaman *funciones
      universales*, o ufuncs.

    * ``np.abs`` no es el ``abs`` incorporado.  El incorporado da la casualidad
      de que funciona con arreglos (delega en NumPy), pero no funciona con
      listas, y ``math.fabs`` falla de plano con arreglos.  Usa la versión
      ``np.`` durante todo este curso.

    * El residuo ``temps - temps_modelo`` tiene signo, y el signo es
      información: dice si la lectura fue más caliente o más fría de lo
      predicho.  Toma el valor absoluto solo cuando quieras un *tamaño*, que es
      lo que pregunta "cuánto se desvía".  La Tarea 17 vuelve a los signos a
      propósito.

    * Si los dos arreglos tienen longitudes distintas, NumPy lanza un
      ``ValueError`` sobre operandos que no se pudieron difundir juntos
      (*broadcast*).  Ese mensaje significa que las formas no encajan, y conviene
      reconocerlo a simple vista.
    """
    # Problema 13: diferencia elemento a elemento y después valor absoluto.
    temps = np.asarray(temps, dtype=float)
    temps_modelo = np.asarray(temps_modelo, dtype=float)
    return np.abs(temps - temps_modelo)


# --------------------------------------------------------------------------
# Problema 14 -- np.sum
# --------------------------------------------------------------------------
def suma_de_cuadrados(residuos):
    """La suma de los residuos al cuadrado: un número que resume el ajuste.

    Parámetros
    ----------
    residuos : array_like
        Los residuos, uno por lectura.

    Devuelve
    --------
    float
        ``residuos[0]**2 + residuos[1]**2 + ...``, como un float de Python.

    Notas
    -----
    * ``np.sum(a)`` suma todas las entradas de ``a`` y devuelve un solo número.
      Una operación que convierte así un arreglo en un escalar es una
      *reducción*; ``np.min``, ``np.max``, ``np.mean`` y ``np.prod`` son las
      otras reducciones que encontrarás.

    * Eleva al cuadrado elemento a elemento con ``r**2``, no con un ``r*r``
      escrito en un bucle ni con ``np.square``: los tres coinciden, y ``**2`` es
      lo que escribe todo el mundo.  Después reduce los cuadrados con un solo
      ``np.sum``.

    * Usa ``np.sum``, no el ``sum`` incorporado.  El incorporado recorre en
      Python y suma de a una entrada.  ``np.sum`` corre en código compilado, y en
      arreglos largos suma además en un orden por pares que acumula bastante
      menos error de redondeo.  La Práctica 2 trata por entero de por qué ese
      último punto importa.

    * La suma de cuadrados es la cantidad que minimizan los mínimos cuadrados,
      así que volverá en la Práctica 8 como lo que se hace pequeño, y en esta
      práctica como el número que te dice que el ajuste de la Parte E es bueno.

    * Envuelve la respuesta en ``float(...)`` para que la función devuelva un
      float de Python y no un ``np.float64``.  Se comportan igual en la
      aritmética pero se imprimen distinto, y un ``np.float64`` devuelto dentro
      de una tabla de resultados es una fealdad pequeña que conviene evitar.
    """
    # Problema 14: eleva al cuadrado elemento a elemento y reduce con np.sum.
    residuos = np.asarray(residuos, dtype=float)
    return float(np.sum(residuos**2))


# --------------------------------------------------------------------------
# Problema 15 -- np.max y np.argmax
# --------------------------------------------------------------------------
def peor_residuo(residuos):
    """El residuo más grande y dónde ocurrió.

    Parámetros
    ----------
    residuos : array_like
        Los residuos absolutos, uno por lectura.

    Devuelve
    --------
    valor : float
        La entrada más grande.
    indice : int
        La posición de esa entrada.

    Notas
    -----
    * ``np.max(a)`` devuelve el *valor* más grande de ``a``.  ``np.argmax(a)``
      devuelve la *posición* de ese valor: un índice entero, no el valor mismo.
      Los dos juntos responden "cuánto de malo, y dónde".

    * ``a[np.argmax(a)]`` y ``np.max(a)`` son siempre el mismo número.  El índice
      es la mitad útil, porque es lo que te permite consultar el tiempo
      correspondiente: ``tiempos[np.argmax(residuos)]``.  Localizar el peor error
      es como informan su exactitud las Prácticas 7 y 9.

    * ``np.argmax`` devuelve la posición del **primer** máximo cuando varias
      entradas empatan.  No construyas nada que dependa de cuál elige.

    * No confundas ``np.max`` con ``np.maximum``.  ``np.max(a)`` reduce un
      arreglo a su entrada más grande; ``np.maximum(a, b)`` compara *dos*
      arreglos entrada por entrada y devuelve un arreglo con las mayores.
      Echar mano del equivocado da o un error de formas o, peor, un escalar
      silenciosamente incorrecto.

    * Devuelve una tupla ``(valor, indice)``, convertida a ``float`` y ``int``
      simples.  Python devuelve varios valores con solo separarlos por una coma,
      y quien llama los desempaqueta igual::

          valor, indice = peor_residuo(r)
    """
    # Problema 15: devuelve el residuo más grande y su índice.
    residuos = np.asarray(residuos, dtype=float)
    indice = int(np.argmax(residuos))
    return float(np.max(residuos)), indice


# ==========================================================================
# PARTE 4 -- Selección de datos
# ==========================================================================


# --------------------------------------------------------------------------
# Problema 16 -- rebanado (y máscaras booleanas)
# --------------------------------------------------------------------------
def caidas_sucesivas(temps):
    """El cambio de temperatura entre lecturas consecutivas.

    Parámetros
    ----------
    temps : array_like
        Las temperaturas registradas, en orden.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(len(temps) - 1,)``.  La entrada ``i`` es
        ``temps[i+1] - temps[i]``, así que es negativa siempre que el café se
        haya enfriado entre las dos lecturas.

    Notas
    -----
    * Una *rebanada* ``a[inicio:fin]`` selecciona las entradas desde ``inicio``
      hasta ``fin`` sin incluirlo.  Dejar un lado fuera significa "hasta el
      final": ``a[2:]`` es todo desde el índice 2 en adelante, ``a[:5]`` las
      primeras cinco entradas.  Un índice negativo cuenta desde el final, así
      que ``a[:-1]`` es todo menos la última entrada.

    * El *rebanado desplazado* es el modismo del que trata esta tarea.
      ``temps[1:]`` y ``temps[:-1]`` tienen ambos ``n - 1`` entradas, y quedan
      alineados de modo que la posición ``i`` del primero cae junto a la
      posición ``i`` del segundo::

          temps[1:]  - temps[:-1]

      son todas las diferencias consecutivas, sin bucle.  Vale la pena escribir
      las dos rebanadas a mano una vez para un arreglo de cuatro entradas y ver
      por qué funciona.

    * El resultado es una entrada más corto que la entrada.  Ese desfase de uno
      es inherente -- ``n`` lecturas tienen ``n - 1`` huecos entre ellas -- y
      olvidarlo es como se producen los errores de índice de la Tarea 18.

    * Una rebanada de un arreglo de NumPy es una *vista*, no una copia:
      comparte memoria con el original, así que escribir en la rebanada cambia
      el original.  (Una rebanada de una *lista* sí es una copia.  Los dos se
      comportan distinto, lo cual sorprende a todo el mundo una vez.)  La resta
      de arriba crea un arreglo nuevo, así que aquí no hay nada en riesgo.

    * La otra manera de seleccionar es una *máscara booleana*.  Comparar un
      arreglo con un número da un arreglo de ``True``/``False`` de la misma
      forma, e indexar con él conserva las entradas donde vale ``True``::

          calientes = temps[temps > 65.0]

      Combina máscaras con ``&`` (y), ``|`` (o) y ``~`` (no), nunca con las
      palabras ``and``/``or``, que intentan reducir todo un arreglo a un único
      valor de verdad y lanzan un ``ValueError``.  Prueba la línea de arriba en
      el intérprete; la Tarea 18 usa la forma con índices de la misma idea.
    """
    # Problema 16: diferencias consecutivas por rebanado desplazado.
    temps = np.asarray(temps, dtype=float)
    return temps[1:] - temps[:-1]


# --------------------------------------------------------------------------
# Problema 17 -- np.sign
# --------------------------------------------------------------------------
def signos_de_caida(caidas):
    """El signo de cada cambio: -1 enfriamiento, +1 calentamiento, 0 sin cambio.

    Parámetros
    ----------
    caidas : array_like
        Cambios consecutivos de temperatura, como los devuelve
        :func:`caidas_sucesivas`.

    Devuelve
    --------
    numpy.ndarray
        La misma forma que ``caidas``, con entradas tomadas de ``-1.0``, ``0.0``
        y ``+1.0``.

    Notas
    -----
    * ``np.sign(a)`` lleva cada entrada a ``-1`` si es negativa, ``+1`` si es
      positiva y ``0`` si es exactamente cero.  Es elemento a elemento, así que
      el resultado tiene la misma forma que la entrada.

    * Físicamente, una taza que se enfría no da más que ``-1``.  Cualquier
      ``+1`` significa que el café se puso *más caliente* entre dos lecturas, lo
      que no puede ocurrir en una habitación más fría que el café: es entonces
      un fallo de medición, no una temperatura.  Esto es una *prueba de
      sensatez*: una comprobación que un resultado debe pasar para ser siquiera
      físicamente posible.  Hazle una a todo conjunto de datos que te entreguen.

    * Probar el signo es más seguro que probar el valor.  Prácticas posteriores
      comparan dos valores de una función para ver si encierran una raíz, y el
      producto de dos valores grandes se desborda a infinito mientras que el de
      dos diminutos se hunde a cero.  El producto de sus *signos* no puede hacer
      ninguna de las dos cosas.  Esa es la prueba de encierro de las Prácticas 4
      y 5, y aquí escribes su primera mitad.

    * El caso del cero exacto es real y muerde: ``np.sign(0.0)`` es ``0.0``, así
      que una prueba estricta como ``signos > 0`` no se dispara con una lectura
      sin cambio.  Decide a propósito si "sin cambio" cuenta como sospechoso
      para tu fin.  Para esta práctica no cuenta.
    """
    # Problema 17: signo elemento a elemento de las caídas.
    caidas = np.asarray(caidas, dtype=float)
    return np.sign(caidas)


# --------------------------------------------------------------------------
# Problema 18 -- np.flatnonzero
# --------------------------------------------------------------------------
def indices_sospechosos(caidas):
    """Índices de las lecturas más calientes que la lectura anterior.

    Parámetros
    ----------
    caidas : array_like
        Cambios consecutivos de temperatura, como los devuelve
        :func:`caidas_sucesivas`.

    Devuelve
    --------
    numpy.ndarray
        Un arreglo unidimensional de enteros, dtype ``int64``.  Cada entrada es
        un índice del **registro original de temperaturas**: si ``caidas[i] > 0``
        entonces la lectura sospechosa es la ``i + 1``.  Vacío, con forma
        ``(0,)``, cuando el registro está limpio.

    Notas
    -----
    * ``np.flatnonzero(a)`` devuelve los índices de las entradas de ``a`` que no
      son cero.  Pásale una máscara booleana y ``True`` cuenta como distinto de
      cero, así que obtienes las posiciones donde se cumplió la condición.

    * Máscara frente a índices, y cuándo quieres cada uno:

      - ``temps[mascara]``            -> los *valores* que pasaron la prueba;
      - ``np.flatnonzero(mascara)``   -> las *posiciones* donde pasó.

      Usa los índices siempre que la posición sea en sí la respuesta, o cuando
      necesites buscar esa misma posición en un segundo arreglo: aquí, para
      encontrar el *tiempo* de la lectura defectuosa en ``TIEMPOS_REG``.

    * El tipo devuelto es un arreglo de enteros, ``int64``, no de booleanos.  Se
      puede usar directamente como índice, y ``len(...)`` sobre él cuenta los
      aciertos.  Un resultado vacío es normal y no debe tratarse como error;
      protégete con ``if idx.size > 0:`` antes de ir por ``idx[0]``.

    * **Cuidado con el desfase de uno.**  ``caidas`` es una entrada más corto
      que ``temps``, y ``caidas[i]`` compara las lecturas ``i`` e ``i + 1``.  Una
      caída positiva en ``i`` acusa por tanto a la lectura ``i + 1``, así que
      súmale 1 a los índices antes de devolverlos.  Equivocarse aquí señala con
      el dedo al vecino inocente.

    * ``np.nonzero`` hace el mismo trabajo pero devuelve una *tupla* de arreglos
      de índices, uno por dimensión, así que para un arreglo 1-D tendrías que
      escribir ``np.nonzero(mascara)[0]``.  ``np.flatnonzero`` es esa expresión
      con el ``[0]`` final ya incorporado, y por eso es la que se usa aquí.
    """
    # Problema 18: índices (en el registro de temperaturas) de los pasos que calientan.
    caidas = np.asarray(caidas, dtype=float)
    return np.flatnonzero(caidas > 0) + 1


# ==========================================================================
# PARTE 5 -- Recuperación de la tasa de enfriamiento
# ==========================================================================


# --------------------------------------------------------------------------
# Problema 19 -- np.column_stack
# --------------------------------------------------------------------------
def matriz_diseno(tiempos):
    """La matriz de dos columnas que describe el modelo de recta.

    Parámetros
    ----------
    tiempos : array_like
        Los tiempos de medición, ``n`` de ellos.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(n, 2)``.  La columna 0 son todo unos, la columna 1 es
        ``tiempos``.

    Notas
    -----
    * ``np.column_stack((a, b))`` pone los arreglos unidimensionales ``a`` y
      ``b`` uno al lado del otro como columnas de una matriz.  Dados dos
      arreglos de longitud ``n`` devuelve un arreglo ``(n, 2)``.  Fíjate en los
      paréntesis: los arreglos se pasan como una sola tupla, no como argumentos
      separados.

    * Por qué esta matriz en concreto.  El modelo es ``z = c0*1 + c1*t``.  La
      fila ``i`` de la matriz guarda las dos cosas que multiplican a las
      incógnitas para la medición ``i``, o sea ``1`` y ``tiempos[i]``.  Apila las
      incógnitas como ``c = [c0, c1]`` y las ``n`` ecuaciones de golpe son
      simplemente ``M @ c = z``.  Una matriz construida así se llama *matriz de
      diseño*, y la Práctica 8 trata de poco más.

    * Las filas son mediciones, las columnas son incógnitas.  Una matriz alta y
      estrecha -- 21 filas, 2 columnas -- significa más mediciones que
      incógnitas, que es la situación normal y la razón de que haga falta un
      ajuste por *mínimos cuadrados* y no una resolución exacta.

    * Construye la columna de unos con :func:`columna_de_unos` de la Tarea 12,
      para que coincida automáticamente con la longitud y el dtype de
      ``tiempos``.

    * Los parientes son ``np.row_stack`` / ``np.vstack``, que apilan arreglos
      como filas, y ``np.hstack``, que concatena a lo largo del eje existente.
      Con entradas 1-D, ``np.hstack`` las pega una tras otra en un solo vector
      largo en vez de hacer columnas: una confusión que produce un arreglo
      ``(2n,)`` donde esperabas ``(n, 2)``.
    """
    # Problema 19: apila una columna de unos junto a los tiempos.
    tiempos = np.asarray(tiempos, dtype=float)
    return np.column_stack((columna_de_unos(tiempos), tiempos))


# --------------------------------------------------------------------------
# Problema 20 -- el operador @
# --------------------------------------------------------------------------
def ecuaciones_normales(M, z):
    """Forma el sistema dos por dos cuya solución es la recta que mejor ajusta.

    Parámetros
    ----------
    M : array_like
        La matriz de diseño, forma ``(n, 2)``.
    z : array_like
        Las temperaturas linealizadas, forma ``(n,)``.

    Devuelve
    --------
    izq : numpy.ndarray
        ``M.T @ M``, forma ``(2, 2)``.
    der : numpy.ndarray
        ``M.T @ z``, forma ``(2,)``.

    Notas
    -----
    * ``A @ B`` es **multiplicación matricial**.  ``A * B`` es multiplicación
      elemento a elemento.  Son operaciones distintas que da la casualidad de
      que se escriben casi igual, y confundirlas es la errata más cara del
      Python numérico: para dos matrices cuadradas las dos son legales y solo
      una es la que querías, así que no hay ningún mensaje de error que te
      salve.

    * La regla de formas de ``@``: las dimensiones interiores deben coincidir y
      el resultado se queda con las exteriores.  ``(2, n) @ (n, 2)`` da
      ``(2, 2)``, y ``(2, n) @ (n,)`` da ``(2,)``.  Comprobar las formas en papel
      antes de teclear es la manera más rápida de acertar con una expresión
      matricial.

    * ``M.T`` es la transpuesta: las filas pasan a ser columnas, así que una
      matriz ``(n, 2)`` pasa a ser ``(2, n)``.  Es un atributo, no un método: sin
      paréntesis.  Igual que una rebanada, es una vista sobre la misma memoria y
      no una copia.

    * Un arreglo 1-D se trata como fila o como columna, según lo que haga legal
      el producto, así que nunca hace falta remodelar ``z``.  Esa comodidad es
      también la razón de que ``M.T @ z`` salga con forma ``(2,)`` y no
      ``(2, 1)``.

    * De dónde salen las ecuaciones.  Con más mediciones que incógnitas,
      ``M @ c = z`` no tiene solución exacta.  La elección de ``c`` que minimiza
      la suma de residuos al cuadrado es en cambio la solución de las
      *ecuaciones normales* ``(M.T @ M) c = M.T @ z``, que es cuadrada y sí
      tiene una.  La Práctica 8 lo deduce como es debido, y muestra además por
      qué formar ``M.T @ M`` no siempre es la mejor manera de resolverlo.
    """
    # Problema 20: construye M.T @ M y M.T @ z.
    M = np.asarray(M, dtype=float)
    z = np.asarray(z, dtype=float)
    return M.T @ M, M.T @ z


# --------------------------------------------------------------------------
# Problema 21 -- np.linalg.solve
# --------------------------------------------------------------------------
def resolver_ecuaciones_normales(izq, der):
    """Resuelve el sistema dos por dos para los coeficientes de la recta.

    Parámetros
    ----------
    izq : array_like
        La matriz cuadrada, forma ``(2, 2)``.
    der : array_like
        El lado derecho, forma ``(2,)``.

    Devuelve
    --------
    numpy.ndarray
        Forma ``(2,)``: los coeficientes ``[c0, c1]`` de ``z = c0 + c1*t``.  La
        tasa de enfriamiento es ``k = -c1``.

    Notas
    -----
    * ``np.linalg.solve(A, b)`` resuelve ``A x = b`` y devuelve ``x``.  ``A``
      debe ser cuadrada y no singular; ``b`` puede ser un vector o una matriz con
      varios lados derechos.

    * **Nunca escribas ``np.linalg.inv(A) @ b``.**  Calcula más de lo que
      pediste, cuesta como tres veces más y es mediblemente menos exacto.
      ``np.linalg.solve`` factoriza la matriz y sustituye, que es lo que en
      realidad es resolver un sistema lineal.  La regla vale en cualquier
      tamaño, y vale aquí en 2 por 2 aunque no se vea salir nada mal.

    * Una matriz singular lanza ``np.linalg.LinAlgError``.  Significa que las
      columnas de tu matriz de diseño no eran independientes: por ejemplo si
      todas las mediciones se hubieran tomado en el mismo instante, con lo que
      la columna de unos y la de tiempos serían proporcionales.

    * Que vuelva una solución no la hace automáticamente buena.  Si la matriz
      está *mal condicionada*, la respuesta puede estar muy equivocada mientras
      todos los residuos parecen pequeños.  ``np.linalg.cond(A)`` mide el
      riesgo; las Prácticas 3 y 8 están construidas alrededor de lo que pasa
      cuando ese número es grande.

    * La Práctica 3 sustituye esta resolución directa por métodos iterativos,
      que es lo que se usa cuando la matriz es demasiado grande para
      factorizarla.
    """
    # Problema 21: resuelve el sistema 2x2.
    return np.linalg.solve(izq, der)


# --------------------------------------------------------------------------
# Problema 22 -- np.linalg.norm
# --------------------------------------------------------------------------
def norma_del_residuo(M, c, z):
    """Qué tan bien reproduce la recta ajustada las mediciones linealizadas.

    Parámetros
    ----------
    M : array_like
        La matriz de diseño, forma ``(n, 2)``.
    c : array_like
        Los coeficientes ajustados, forma ``(2,)``.
    z : array_like
        Las temperaturas linealizadas, forma ``(n,)``.

    Devuelve
    --------
    float
        La norma euclidiana ``||M @ c - z||_2``.

    Notas
    -----
    * ``np.linalg.norm(v)`` devuelve la longitud euclidiana del vector ``v``, es
      decir ``sqrt(sum(v**2))``.  Es una reducción: entra un arreglo, sale un
      número.

    * El segundo argumento opcional elige otra norma.  ``ord=1`` suma los
      valores absolutos, ``ord=np.inf`` toma el mayor valor absoluto.  El valor
      por omisión, ``ord=2``, es la euclidiana y es la que pide esta tarea.  La
      Práctica 3 mide la convergencia con ``np.inf``, porque "cada entrada está
      dentro de tol" es una promesa más fuerte que "el total es pequeño".

    * Calcula primero el *vector* de residuos ``M @ c - z`` y después su norma.
      Su longitud es ``n``, una entrada por medición, y vale la pena imprimirlo
      una vez: los mínimos cuadrados hacen pequeño el total, no cada entrada.

    * Una norma del residuo mide el *ajuste*, no la *corrección*.  Dice que la
      recta pasa cerca de los datos; no dice nada sobre si una recta era el
      modelo adecuado.  Mira siempre los residuos además de su norma; la Tarea
      26 es la gráfica que lo hace.

    * Devuelve ``float(...)``, por la razón dada en la Tarea 14.
    """
    # Problema 22: vector de residuos y después su norma 2.
    residuo = M @ c - z
    return float(np.linalg.norm(residuo))


# --------------------------------------------------------------------------
# Problema 23 -- np.polyval
# --------------------------------------------------------------------------
def evaluar_recta(c, tiempos):
    """Evalúa la recta ajustada ``z = c0 + c1*t`` en tiempos arbitrarios.

    Parámetros
    ----------
    c : array_like
        Los coeficientes ``[c0, c1]`` en potencias *crecientes* de ``t``,
        exactamente como los devuelve :func:`resolver_ecuaciones_normales`.
    tiempos : array_like
        Dónde evaluar la recta.

    Devuelve
    --------
    numpy.ndarray
        La misma forma que ``tiempos``.

    Notas
    -----
    * ``np.polyval(p, x)`` evalúa en cada punto de ``x`` el polinomio cuyos
      coeficientes son ``p``, y devuelve un arreglo con la forma de ``x``.

    * **El orden de los coeficientes es la trampa.**  ``np.polyval`` espera
      potencias *decrecientes*: ``p = [3, 2, 1]`` significa ``3x^2 + 2x + 1``,
      con la potencia más alta primero.  Tu ``c`` de la Tarea 21 está en
      potencias crecientes, ``c0 + c1*t``, así que tienes que invertirlo antes de
      pasarlo: ``c[::-1]`` es la rebanada que lo hace, y es la notación de
      rebanado desplazado de la Tarea 16 con paso -1.  Dáselo al revés y
      ajustarás en silencio ``c1 + c0*t``, que es una recta completamente
      distinta, sin ningún mensaje de error por ninguna parte.

    * ``np.polyval`` evalúa por la regla de Horner: multiplicaciones y sumas
      encadenadas en lugar de calcular cada potencia por separado.  Eso es a la
      vez más rápido y más exacto que la fórmula obvia, y en la Práctica 6 la
      escribirás tú como división sintética.

    * ``np.polyfit(x, y, grado)`` es la función hermana que *encuentra* los
      coeficientes, en el mismo orden decreciente.  Haría todas las Tareas 19 a
      21 en una sola llamada.  Aquí construyes el ajuste a mano porque la
      Práctica 8 necesita que sepas qué hace esa llamada.

    * La clase moderna ``np.polynomial.Polynomial`` usa en cambio el orden
      *creciente*.  Dos convenios conviven en NumPy; comprueba siempre cuál
      quiere cada función.
    """
    # Problema 23: evalúa con np.polyval, cuidando el orden de los coeficientes.
    c = np.asarray(c, dtype=float)
    tiempos = np.asarray(tiempos, dtype=float)
    return np.polyval(c[::-1], tiempos)


# ==========================================================================
# PARTE 6 -- Dibujo de gráficas
# ==========================================================================

# --------------------------------------------------------------------------
# Problema 24 -- ax.plot
# --------------------------------------------------------------------------
def graficar_registro_y_modelo(tiempos, temps, ax=None):
    """Dibuja las lecturas registradas y la curva del modelo en unos mismos ejes.

    Parámetros
    ----------
    tiempos : array_like
        Los tiempos de medición, en minutos.
    temps : array_like
        Las temperaturas registradas, en grados Celsius.
    ax : matplotlib.axes.Axes o None
        Los ejes sobre los que dibujar.  Si es ``None``, crea una figura y unos
        ejes nuevos.

    Devuelve
    --------
    matplotlib.axes.Axes
        Los ejes sobre los que se dibujó.

    Notas
    -----
    * Matplotlib tiene dos interfaces.  ``plt.plot(...)`` dibuja sobre la figura
      que resulte ser la actual, lo cual es cómodo en el intérprete e inservible
      dentro de una función.  Todo en este curso usa en cambio la forma
      explícita::

          fig, ax = plt.subplots()
          ax.plot(x, y, label="mi curva")

      ``plt.subplots()`` devuelve un par ``(figura, ejes)``: la figura es el
      dibujo entero, los ejes son la caja con los datos dentro.

    * ``ax.plot(x, y, fmt, **kwargs)`` dibuja ``y`` frente a ``x``.  Los dos
      deben tener la misma longitud; longitudes distintas son el error más común
      aquí, y la causa habitual es el arreglo una entrada más corto de la Tarea
      16.  La cadena de formato opcional combina un marcador, un estilo de línea
      y un color: ``"o"`` son solo puntos, ``"-"`` una línea simple, ``"o-"``
      ambos, ``"k--"`` una línea negra a trazos.  Después van palabras clave
      como ``label=``, ``linewidth=`` y ``color=``.

    * Las mediciones son puntos y un modelo es una curva.  Dibuja el registro con
      ``"o"`` y el modelo con una línea simple sobre una malla mucho más fina;
      usa :func:`malla_fina_de_tiempo` de la Tarea 7, no los 21 tiempos de
      medición, o tu "curva" serán 20 segmentos rectos.

    * Una etiqueta solo aparece si pides la leyenda: pasa ``label=`` a cada
      llamada de ``ax.plot`` y después llama una vez a ``ax.legend()``.  Rotula
      ambos ejes con ``ax.set_xlabel`` / ``ax.set_ylabel``, unidades incluidas, y
      marca el umbral de bebible con ``ax.axhline(OBJETIVO, ...)``.

    * Acepta un argumento ``ax`` opcional, créalo solo cuando sea ``None``, y
      **devuelve los ejes**.  Eso es lo que permite a quien llama dibujar varias
      curvas en una figura o añadir anotaciones sobre las tuyas.  Todas las
      funciones de dibujo de este curso siguen este convenio, y la prueba de esta
      tarea comprueba que el objeto que devuelves es el que te dieron.
    """
    # Problema 25: dibuja las lecturas como puntos y el modelo como curva.
    tiempos = np.asarray(tiempos, dtype=float)
    temps = np.asarray(temps, dtype=float)
    if ax is None:
        _, ax = plt.subplots()
    fina = malla_fina_de_tiempo(float(np.max(tiempos)))
    ax.plot(fina, modelo(fina), "-", label="ley de Newton")
    ax.plot(tiempos, temps, "o", markersize=4, label="registrador")
    ax.axhline(OBJETIVO, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("tiempo t (min)")
    ax.set_ylabel("temperatura T (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


# ==========================================================================
# Programa principal -- ejecútelo para producir los resultados.
#
# Puede agregar funcionalidad especificando el problema. 
# ==========================================================================
def main():
    linea = "=" * 66

    print(linea)
    print("PARTE 1 -- Bucles en Python y listas")
    print(linea)
    n_min = minutos_hasta_bebible()
    print(f"el modelo llega a {OBJETIVO:.0f} C tras {n_min} minutos enteros")
    idx = primera_por_debajo(TEMPS_REG)
    print(f"la primera lectura igual o menor que {OBJETIVO:.0f} C es la número {idx}"
          f" (t = {TIEMPOS_REG[idx]:.0f} min)")
    por_append = simular_con_append()
    por_compr = simular_por_comprension()
    print(f"simulados {len(por_append)} minutos; append y comprensión coinciden:"
          f" {por_append == por_compr}")

    print()
    print(linea)
    print("PARTE 2 -- Arreglos con Numpy")
    print(linea)
    temps = registro_a_arreglo(TEMPS_REG)
    tiempos = registro_a_arreglo(TIEMPOS_REG)
    print(f"temps.shape = {temps.shape}, temps.dtype = {temps.dtype}")
    print(f"como_arreglo_float devuelve el mismo objeto: "
          f"{como_arreglo_float(temps) is temps}")
    fina = malla_fina_de_tiempo()
    gruesa = malla_gruesa_de_tiempo()
    print(f"malla fina:   {fina.size} puntos, h = {fina[1] - fina[0]:.3f} min")
    print(f"malla gruesa: {gruesa}")
    print(f"columna_constante(3, 7.0) = {columna_constante(3, 7.0)}")
    modelo_ajustado = columna_del_modelo(tiempos)
    print(f"modelo en los 21 tiempos del registro, primeros tres: "
          f"{np.array2string(modelo_ajustado[:3], precision=3)}")
    print(f"contenedor_de_residuos(temps).shape = "
          f"{contenedor_de_residuos(temps).shape}")
    print(f"columna_de_unos(tiempos)[:4] = {columna_de_unos(tiempos)[:4]}")

    print()
    print(linea)
    print("PARTE 3 -- Operaciones elemento a elemento y reducciones")
    print(linea)
    resid = residuos_absolutos(temps, modelo_ajustado)
    print(f"suma de residuos al cuadrado = {suma_de_cuadrados(resid):.4f}")
    peor, donde = peor_residuo(resid)
    print(f"peor residuo = {peor:.3f} C en la lectura {donde} "
          f"(t = {tiempos[donde]:.0f} min)")

    print()
    print(linea)
    print("PARTE 4 -- Selección de datos")
    print(linea)
    caidas = caidas_sucesivas(temps)
    print(f"{caidas.size} caídas para {temps.size} lecturas")
    signos = signos_de_caida(caidas)
    print(f"número de pasos que no enfrían = {int(np.sum(signos > 0))}")
    malas = indices_sospechosos(caidas)
    for i in malas:
        print(f"la lectura {i} en t = {tiempos[i]:.0f} min marca {temps[i]:.1f} C, "
              f"más que los {temps[i - 1]:.1f} C anteriores")

    print()
    print(linea)
    print("PARTE 5 -- Recuperación de la tasa de enfriamiento")
    print(linea)
    z = linealizar(temps)
    M = matriz_diseno(tiempos)
    izq, der = ecuaciones_normales(M, z)
    print(f"M.shape = {M.shape}, (M.T @ M).shape = {izq.shape}")
    c = resolver_ecuaciones_normales(izq, der)
    k_encontrada = -c[1]
    print(f"recta ajustada: z = {c[0]:.6f} {c[1]:+.6f} t")
    print(f"k recuperada = {k_encontrada:.6f} 1/min   (valor real {K_REAL})")
    print(f"T0 - T_amb recuperado = {np.exp(c[0]):.4f} C   "
          f"(valor real {T0 - T_AMB})")
    print(f"norma del residuo ||M c - z|| = {norma_del_residuo(M, c, z):.6f}")
    print(f"la recta en t = 0, 20, 40: "
          f"{np.array2string(evaluar_recta(c, np.array([0.0, 20.0, 40.0])), precision=4)}")

    print()
    print(linea)
    print("PARTE 6 -- Dibujo de gráficas")
    print(linea)
    print(f"k_encontrada == K_REAL       -> {k_encontrada == K_REAL}")

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    graficar_registro_y_modelo(tiempos, temps, ax=ejes[0])
    fig.tight_layout()
    fig.savefig("lab01_cafe.png", dpi=150, bbox_inches="tight")
    print("Figura escrita: lab01_cafe.png")


if __name__ == "__main__":
    main()
