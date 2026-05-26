"""
Módulo: semantic_memory
Autor: Manuel David Alcántara (IBM Corp.)

Descripción:
Comprime el historial de conversación (conversation_history) antes de enviarlo
al LLM principal, reduciendo tokens sin perder contexto semántico relevante.

Estrategia:
- Preserva SIEMPRE los últimos 2 turnos completos (ASISTENTE + CLIENTE)
- Resume los mensajes INTERMEDIOS del ASISTENTE que superen el umbral de longitud
- Mantiene SIEMPRE los mensajes del CLIENTE intactos (son cortos y contienen la intención)
- Solo activa compresión cuando numero_msg_cliente >= MIN_TURNS_TO_COMPRESS

Formato del historial:
    "ASISTENTE VIRTUAL: 'msg1' | CLIENTE: 'msg2' | ASISTENTE VIRTUAL: 'msg3' | CLIENTE: 'msg4'"
"""

import re
import logging
from typing import Optional


# ── Parser del historial ──────────────────────────────────────────────────────

def parse_conversation(conversation_history: str) -> list[dict]:
    """
    Convierte el string de historial en una lista de turnos.

    Returns:
        Lista de dicts con keys: 'role' ('ASISTENTE VIRTUAL' o 'CLIENTE'), 'message'
    """
    turnos = []
    # Separar por el delimitador " | "
    partes = re.split(r'\s*\|\s*', conversation_history.strip())

    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue

        match_a = re.match(r"ASISTENTE VIRTUAL:\s*'(.*)'", parte, re.DOTALL)
        match_c = re.match(r"CLIENTE:\s*'(.*)'", parte, re.DOTALL)

        if match_a:
            turnos.append({'role': 'ASISTENTE VIRTUAL', 'message': match_a.group(1).strip()})
        elif match_c:
            turnos.append({'role': 'CLIENTE', 'message': match_c.group(1).strip()})

    return turnos


def build_conversation(turnos: list[dict]) -> str:
    """
    Reconstruye el string de historial desde la lista de turnos.
    """
    partes = []
    for t in turnos:
        partes.append(f"{t['role']}: '{t['message']}'")
    return ' | '.join(partes)


# ── Función de resumen de un turno del asistente ──────────────────────────────

def _resumir_turno_asistente(mensaje: str, max_asistente_chars: int) -> str:
    """
    Resume un mensaje largo del ASISTENTE VIRTUAL preservando el inicio y el final.

    Estrategia:
    - Toma el 50% de MAX_ASISTENTE_CHARS del inicio del mensaje
    - Toma el 50% de MAX_ASISTENTE_CHARS del final del mensaje
    - Une ambas partes con "..."
    - Esto preserva el contexto inicial Y la pregunta de cierre del asistente
    """
    msg = mensaje.strip()

    if len(msg) <= max_asistente_chars:
        return msg

    mitad = max_asistente_chars // 2
    inicio = msg[:mitad].rstrip()
    final  = msg[-mitad:].lstrip()

    return f"{inicio}...{final}"


# ── Función principal ─────────────────────────────────────────────────────────

def compress_conversation_history(
    conversation_history: str,
    min_turns_to_compress: int = 3,
    max_asistente_chars: int = 24,
    preserve_last_n_turns: int = 1,
    numero_msg_cliente: Optional[int] = None,
) -> str:
    # ── Configuración ─────────────────────────────────────────────────────────────

    # Activar compresión solo cuando el cliente ha enviado este número de mensajes o más
    #MIN_TURNS_TO_COMPRESS: int = 3

    # Resumir mensajes del asistente que superen este número de caracteres
    #MAX_ASISTENTE_CHARS: int = 200
    #MAX_ASISTENTE_CHARS: int = 24

    # Número de turnos finales (ASISTENTE + CLIENTE) que se preservan completos
    #PRESERVE_LAST_N_TURNS: int = 2
    #PRESERVE_LAST_N_TURNS: int = 1 #NO BAJAR DE 1 - Con valor=1 mantiene intacto el último turno completo (ASISTENTE + CLIENTE)
    #PRESERVE_LAST_N_TURNS: int = 0 #Con valor=0 no preserva ningún turno completo, pero sí preserva íntegros los mensajes del cliente y resume los del asistente según el umbral de caracteres

    """
    Comprime el historial de conversación preservando el contexto semántico.

    Args:
        conversation_history: String con el historial en formato estándar.
        numero_msg_cliente:   Número de mensajes del cliente (opcional).
                              Si se proporciona, evita recalcularlo internamente.

    Returns:
        String con el historial comprimido (o el original si no aplica compresión).
    """

    if not conversation_history:
        return conversation_history

    # ── 1. Calcular número de mensajes del cliente ────────────────────────────
    if numero_msg_cliente is None:
        numero_msg_cliente = len(re.findall(r'\bCLIENTE\s*:', conversation_history))

    # ── 2. Decidir si comprimir ───────────────────────────────────────────────
    if numero_msg_cliente < min_turns_to_compress:
        logging.debug(
            f"[semantic_memory] Sin compresión: {numero_msg_cliente} msgs cliente "
            f"< umbral {min_turns_to_compress}"
        )
        return conversation_history

    # ── 3. Parsear historial ──────────────────────────────────────────────────
    turnos = parse_conversation(conversation_history)
    if not turnos:
        return conversation_history

    total = len(turnos)

    # ── 4. Identificar los últimos N turnos a preservar ───────────────────────
    # Un "turno" = par ASISTENTE + CLIENTE
    # Contamos desde el final cuántos índices preservar
    turnos_pares_a_preservar = preserve_last_n_turns * 2  # cada turno tiene 2 mensajes
    inicio_preservar = max(0, total - turnos_pares_a_preservar)

    # ── 5. Comprimir turnos intermedios del ASISTENTE ─────────────────────────
    turnos_comprimidos = []
    for i, turno in enumerate(turnos):
        if i >= inicio_preservar:
            # Turno reciente → preservar completo
            turnos_comprimidos.append(turno)
        elif turno['role'] == 'ASISTENTE VIRTUAL':
            # Turno intermedio del asistente → comprimir si es largo
            if len(turno['message']) > max_asistente_chars:
                resumen = _resumir_turno_asistente(turno['message'], max_asistente_chars)
                turnos_comprimidos.append({'role': 'ASISTENTE VIRTUAL', 'message': resumen})
                logging.debug(
                    f"[semantic_memory] Turno {i} comprimido: "
                    f"{len(turno['message'])} → {len(resumen)} chars"
                )
            else:
                turnos_comprimidos.append(turno)
        else:
            # Turno del CLIENTE → siempre preservar completo
            turnos_comprimidos.append(turno)

    # ── 6. Reconstruir y retornar ─────────────────────────────────────────────
    historia_comprimida = build_conversation(turnos_comprimidos)

    original_len   = len(conversation_history)
    comprimida_len = len(historia_comprimida)
    reduccion_pct  = (1 - comprimida_len / original_len) * 100 if original_len > 0 else 0

    logging.info(
        f"[semantic_memory] Compresión aplicada: "
        f"{original_len} → {comprimida_len} chars "
        f"({reduccion_pct:.1f}% reducción)"
    )

    return historia_comprimida


# ── Truncado por duplas ───────────────────────────────────────────────────────

def truncate_to_last_duplas(
    conversation_history: str,
    max_dupla_msgs: int = 2,
) -> str:
    """
    Trunca el historial conservando solo las últimas N duplas (ASISTENTE + CLIENTE).
    Las duplas anteriores se reemplazan por "...".

    Args:
        conversation_history: String con el historial en formato estándar.
        max_dupla_msgs:        Número de duplas a conservar desde el final.

    Returns:
        Historial truncado, o el original si el total de duplas <= max_dupla_msgs.

    Ejemplo (max_dupla_msgs=1):
        "ASISTENTE: 'A' | CLIENTE: 'B' | ASISTENTE: 'C' | CLIENTE: 'D'"
        → "... | ASISTENTE: 'C' | CLIENTE: 'D'"
    """
    if not conversation_history or max_dupla_msgs <= 0:
        return conversation_history

    turnos = parse_conversation(conversation_history)
    if not turnos:
        return conversation_history

    # Agrupar turnos en duplas (ASISTENTE, CLIENTE)
    duplas = []
    i = 0
    while i < len(turnos):
        if turnos[i]['role'] == 'ASISTENTE VIRTUAL' and i + 1 < len(turnos) and turnos[i + 1]['role'] == 'CLIENTE':
            duplas.append((turnos[i], turnos[i + 1]))
            i += 2
        else:
            # Turno suelto (CLIENTE sin ASISTENTE previo, o ASISTENTE al final sin CLIENTE)
            duplas.append((None, turnos[i]) if turnos[i]['role'] == 'CLIENTE' else (turnos[i], None))
            i += 1

    if len(duplas) <= max_dupla_msgs:
        return conversation_history

    # Reconstruir solo las últimas N duplas
    partes = ["..."]
    for asistente, cliente in duplas[-max_dupla_msgs:]:
        if asistente:
            partes.append(f"ASISTENTE VIRTUAL: '{asistente['message']}'")
        if cliente:
            partes.append(f"CLIENTE: '{cliente['message']}'")

    historia_truncada = " | ".join(partes)

    logging.info(
        f"[semantic_memory] Truncado aplicado: "
        f"{len(duplas)} duplas → {max_dupla_msgs} duplas "
        f"({len(conversation_history)} → {len(historia_truncada)} chars)"
    )

    return historia_truncada