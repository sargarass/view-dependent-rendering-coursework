#include "Shader.h"
#include "bothInclude.h"
#include <fstream>

Shader::Shader() {
    this->m_fragmentShader = this->m_program = this->m_vertexShader = m_load = 0;
}

Shader::~Shader() {
    clear();
}

void Shader::clear() {
    if (m_program) {
        glDeleteProgram(m_program);
    }
    if (m_vertexShader) {
        glDeleteShader(m_vertexShader);
    }
    if (m_fragmentShader) {
        glDeleteShader(m_fragmentShader);
    }
    m_fragmentShader = 0;
    m_program = 0;
    m_vertexShader = 0;
    m_load = 0;
    m_uniformMap.clear();
    m_attribMap.clear();
}

bool Shader::load(std::string const &path_vertex, std::string const &path_frag) {
    std::ifstream fvert(path_vertex);
    std::ifstream ffrag(path_frag);
    if (!fvert.is_open() || !ffrag.is_open()) {
        m_load = false;
        return false;
    }
    clear();


    std::string VertexSource((std::istreambuf_iterator<char>(fvert)),
                     std::istreambuf_iterator<char>());
    std::string FragSource((std::istreambuf_iterator<char>(ffrag)),
                     std::istreambuf_iterator<char>());

    if (VertexSource.empty() || FragSource.empty()) {
        clear();
        return false;
    }

    m_program = glCreateProgram();
    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    if (!m_vertexShader || !m_vertexShader || !m_fragmentShader) {
        clear();
        return false;
    }

    GLchar const *tmp = static_cast<const GLchar*>(VertexSource.c_str());
    glShaderSource(m_vertexShader, 1, static_cast<GLchar const **>(&tmp), NULL);

    tmp = static_cast<const GLchar*>(FragSource.c_str());
    glShaderSource(m_fragmentShader, 1, static_cast<GLchar const **>(&tmp), NULL);

    if (!compileShader(m_vertexShader, path_vertex) || !compileShader(m_fragmentShader, path_frag))
    {
        glDeleteProgram(m_program);
        glDeleteShader(m_vertexShader);
        glDeleteShader(m_fragmentShader);
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Shader", "load", "Could not compile the shaders, they are invalid");
        clear();
        return false;
    }

    glAttachShader(m_program, m_vertexShader);
    glAttachShader(m_program, m_fragmentShader);
    glLinkProgram(m_program);
    GLint status;
    glGetProgramiv(m_program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        std::string infoLog;
        GLint infoLen;

        glGetProgramiv(m_program, GL_INFO_LOG_LENGTH, &infoLen);
        Log::getInstance().write(LOG_MESSAGE_TYPE::WARNING, "Shader", "load", "Linking error!");
        infoLog.resize(infoLen);
        glGetProgramInfoLog(m_program, infoLog.size(), &infoLen, &infoLog[0]);
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Shader", "load", "%s", infoLog.c_str());
        clear();
        return false;
    }
    glUseProgram(0);

    m_load = true;
    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO, "Shader", "load","Status OK");
    return true;
}

bool Shader::compileShader(uint &shader, std::string const &name)
{
    glCompileShader(shader);
    GLint result;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);

    if (result == GL_FALSE)
    {
        std::string infoLog;
        GLint infoLen;

        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
        Log::getInstance().write(LOG_MESSAGE_TYPE::WARNING, "Shader", "compileShader", "Shader \"%s\" contains errors, please validate this shader!", name.c_str());
        infoLog.resize(infoLen);
        glGetShaderInfoLog(shader, infoLog.size(), &infoLen, &infoLog[0]);
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Shader", "compileShader", "%s", infoLog.c_str());
        return false;
    }

    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO, "Shader", "compileShader", "\"%s\" Status: OK\n", name.c_str());
    return true;
}

bool Shader::bind()
{
    if(m_program)
    {
       glUseProgram(m_program);
       return true;
    }
    return false;
}

bool Shader::unbind()
{
    glUseProgram(0);
    return true;
}

uint Shader::getUniformLocation(std::string const &name)
{
    btree::btree_map<std::string, uint>::iterator i = m_uniformMap.find(name);
    if (i == m_uniformMap.end())
    {

        GLuint location = glGetUniformLocation(m_program, name.c_str());
        m_uniformMap.insert(std::make_pair(name, location));
        return location;
    }
    return (*i).second;
}

uint Shader::getAttribLocation(std::string const &name)
{
    btree::btree_map<std::string, uint>::iterator i = m_attribMap.find(name);
    if (i == m_attribMap.end())
    {
        GLuint location = glGetAttribLocation(m_program, name.c_str());
        m_attribMap.insert(std::make_pair(name, location));
        return location;
    }
    return (*i).second;
}

void Shader::setVal(std::string const &name, int const id)
{
    GLuint location = getUniformLocation(name);
    glUniform1i(location, id);
}

void Shader::setVal(std::string const &name, uint const id)
{
    GLuint location = getUniformLocation(name);
    glUniform1i(location, id);
}

void Shader::setVal(std::string const &name, float const id)
{
    GLuint location = getUniformLocation(name);
    glUniform1f(location, id);
}

void Shader::setVal(std::string const &name, double const id)
{
    GLuint location = getUniformLocation(name);
    glUniform1d(location, id);
}

void Shader::setVal(std::string const &name, glm::vec2 const &id)
{
    GLuint location = getUniformLocation(name);
    glUniform2fv(location, sizeof(glm::vec2), glm::value_ptr(id));
}

void Shader::setVal(std::string const &name,  glm::vec3 const &id)
{
    GLuint location = getUniformLocation(name);
    glUniform3f(location, id.x, id.y, id.z);
}

void Shader::setVal(std::string const &name, glm::vec4 const &id)
{
    GLuint location = getUniformLocation(name);
    glUniform4f(location, id.r, id.g, id.b, id.a);
}

void Shader::setVal4x4(std::string const &name, glm::mat4 const &id)
{
    GLuint location = getUniformLocation(name);
    glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(id));
}

void Shader::setVal4x4(std::string const &name, float const *id)
{
    GLuint location = getUniformLocation(name);
    glUniformMatrix4fv(location, 1, GL_FALSE, id);
}

void Shader::setVal3x3(std::string const &name,float const *id)
{
    GLuint location = getUniformLocation(name);
    glUniformMatrix3fv(location, 1, GL_FALSE, id);
}

void Shader::setVal3x3(std::string const &name, glm::mat3 const &id)
{
    GLuint location = getUniformLocation(name);
    glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(id));
}

void Shader::setVal(std::string const &name, float const red, float const green, float const blue, float const alpha)
{
     GLuint location = getUniformLocation(name);
     glUniform4f(location, red, green, blue, alpha);
}

void Shader::setVal(std::string const &name, float const x, float const y, float const z)
{
     GLuint location = getUniformLocation(name);
     glUniform3f(location, x, y, z);
}
