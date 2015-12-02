#pragma once

#include "bothInclude.h"
#include "cpp-btree/btree_map.h"
#include "cpp-btree/btree_set.h"

class Shader {
public:
    bool bind();
    bool unbind();
    void setVal( std::string const &name, int const id);
    void setVal( std::string const &name, uint const id);
    void setVal( std::string const &name, float const id);
    void setVal( std::string const &name, double const id);
    void setVal( std::string const &name, glm::vec2 const &id);
    void setVal( std::string const &name, glm::vec3 const &id);
    void setVal( std::string const &name, glm::vec4 const &id);
    void setVal4x4( std::string const &name, glm::mat4 const &id);
    void setVal4x4( std::string const &name, float const* id);
    void setVal3x3( std::string const &name, float const* id);
    void setVal3x3( std::string const &name, glm::mat3 const &id);
    void setVal(std::string const& name, float const red, float const  green, float const blue, float const alpha);
    void setVal(std::string const& name, float const x, float const y, float  const z);
    bool load(std::string const &pathVertex, std::string const &pathFrag);
    void clear();
    uint getAttribLocation(std::string const &name);
    bool isLoad() const { return m_load; }
    Shader();
    virtual ~Shader();
private:
    bool compileShader(uint &shader, std::string const &name);

    uint getUniformLocation(std::string const &name);
    uint m_program;
    bool m_load;
    uint m_vertexShader;
    uint m_fragmentShader;
    btree::btree_map<std::string, uint> m_uniformMap;
    btree::btree_map<std::string, uint> m_attribMap;
};


struct Material {
    glm::vec4 ambient;
    glm::vec4 diffuse;
    glm::vec4 specular;
    glm::vec4 emission;
    float shininess;

    Material() {
        ambient = {1.0, 1.0, 1.0, 1.0};
        diffuse = {1.0, 1.0, 1.0, 1.0};
        specular = {0.0, 0.0, 0.0, 1.0};
        emission = {0.0, 0.0, 0.0, 1.0};
        shininess = 0.0f;
    }

    void bind(Shader& shader) {
        shader.setVal("material.ambient", ambient);
        shader.setVal("material.diffuse", diffuse);
        shader.setVal("material.specular", specular);
        shader.setVal("material.emission", emission);
        shader.setVal("material.shininess", shininess);
    }
};

struct Light {
    glm::vec4 ambientColor;
    glm::vec4 diffuseColor;
    glm::vec4 specularColor;
    glm::vec3 position;

    Light() {
        position = {0.0, 0.0, 0.0};
        ambientColor = {1.0, 1.0, 1.0, 1.0};
        diffuseColor = ambientColor;
        specularColor = {0.0, 0.0, 0.0, 0.0};
    }

    void bind(Shader& shader) {
        shader.setVal("light.ambientColor", ambientColor);
        shader.setVal("light.diffuseColor", diffuseColor);
        shader.setVal("light.specularColor", specularColor);
        shader.setVal("light.position", position);
    }
};
