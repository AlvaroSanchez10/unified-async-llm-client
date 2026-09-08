# Unified Async LLM Client

Pre-entrega 1 del curso de AI Engineering.

Este proyecto implementa un cliente unificado y asíncrono para trabajar con distintos proveedores de LLM, actualmente OpenAI y Anthropic.

## Funcionalidades

- Interfaz común para OpenAI y Anthropic.
- Uso de programación asíncrona con `async/await`.
- Streaming de respuestas mediante generadores asíncronos.
- Validación de datos utilizando Pydantic.
- Configuración del proveedor mediante variables de entorno.
- Manejo controlado de errores de conexión y rate limiting.
- Soporte para respuestas normales y streaming.

## Requisitos

- Python 3.12 o superior.

## Instalación

Clonar el repositorio:

```bash
git clone URL_DEL_REPOSITORIO
cd unified-async-llm-client
