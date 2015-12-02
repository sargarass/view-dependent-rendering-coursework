#pragma once
#include "bothInclude.h"

class Camera {
public:
    Camera();
    void move(glm::vec3 speed, float deltaTime);
    void rotate(float angle, const glm::vec3& axis);
    void setViewByMouse(int x, int y, float dt);
    void rotatePitch(float angle);
    void rotateYaw(float angle);
    void rotateRoll(float angle);
    void translate(glm::vec3 position);
    glm::mat4 getViewMatrix();
    void lookAt(const glm::vec3& center, const glm::vec3& up);

    void setProjection(double fov, double aspect, double znear, double zfar) {
        m_projectionMatrix = glm::perspective(fov, aspect, znear, zfar);
    }

    glm::mat4 getProjectionMatrix() {
        return m_projectionMatrix;
    }

public:
    glm::quat m_rotation;
    glm::vec3 m_position;

    glm::vec3 m_forward;
    glm::vec3 m_up;

    glm::mat4 m_viewMatrix;
    glm::mat4 m_projectionMatrix;
    bool       m_viewMatrixRecalc;
};
