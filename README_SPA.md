# ChatENP: Un Asistente de Búsqueda Mejorado 🤖

ChatENP es una aplicación de IA conversacional impulsada por Streamlit 💻, OpenAI 🧠, y Pinecone 🌲. Utiliza GPT-3.5-turbo-16k para generar respuestas 💬 y un modelo de QA basado en recuperación para buscar en una base de conocimientos 📚. La base de conocimientos es un conjunto de documentos subidos por el usuario 👤, y la IA utiliza estos documentos para responder consultas 💭❓.

## Empezando 🚀

### Prerrequisitos 📋
- Python 3.7 o superior 🐍.

### Instalación 🔧

1. **Clona este repositorio a tu máquina local 💾:**
    ```
    git clone https://github.com/tunombredeusuario/ChatENP.git
    cd ChatENP
    ```

2. **Crea un entorno virtual ⚙️:**
    ```
    python3 -m venv ChatENP
    source ChatENP/bin/activate  # En Windows, usa `ChatENP\Scripts\activate`
    ```

3. **Instala los paquetes necesarios 📦:**
    ```
    pip install -r requirements.txt
    ```

4. **Configura el archivo de secretos 🔑:**

    Crea un directorio `.streamlit` en la raíz de tu proyecto. Dentro del directorio `.streamlit`, crea un archivo `secrets.toml` y añade tus claves API:
    ```toml
    # secrets.toml
    [secrets]
    OPENAI_API_KEY = "tu-clave-api-openai"
    SERPAPI_API_KEY = "tu-clave-api-serpapi"
    PINECONE_API_KEY = "tu-clave-api-pinecone"
    PINECONE_API_ENV = "tu-ambiente-api-pinecone"
    ```
    Por favor, reemplaza `"tu-clave-api-openai"`, `"tu-clave-api-serpapi"`, `"tu-clave-api-pinecone"`, y `"tu-ambiente-api-pinecone"` con tus claves reales.

5. **Ejecuta la aplicación de Streamlit 💫:**
    ```
    streamlit run ChatENP.py
    ```
6. Navega a la URL local proporcionada en tu navegador web 🌐.

### Usando la Aplicación 📲

- **Sube tus documentos 📤:** En la barra lateral, puedes subir tus documentos PDF que se utilizarán como base de conocimientos para responder a las consultas. Introduce un espacio de nombres de Pinecone y haz clic en "Procesar⚙️". La aplicación extraerá entonces el texto de los PDFs subidos, creará incrustaciones para el texto, y almacenará estas incrustaciones en un índice de Pinecone.
  
- **Elige un espacio de nombres para consultar 🔍:** En la barra lateral, puedes seleccionar el espacio de nombres del cual quieres recuperar información. Los espacios de nombres corresponden a los diferentes conjuntos de documentos que has subido.
  
- **Haz una pregunta 💭❓:** En el panel principal, introduce tu pregunta en el área de texto etiquetada "Consulta🔍" y haz clic en "Enviar📤". La aplicación utilizará el modelo GPT-3.5-turbo para generar una respuesta, consultando el conjunto de documentos cuando sea necesario.
