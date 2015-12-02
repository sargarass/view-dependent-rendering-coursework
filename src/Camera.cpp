#include "Camera.h"

Camera::Camera() {
    m_viewMatrix = glm::mat4(1.0);
    m_viewMatrixRecalc = false;
}

void Camera::move(glm::vec3 speed, float deltaTime) {
    m_position += cross(m_up, -m_forward) * speed.x * deltaTime;
    m_position += m_up * speed.y * deltaTime;
    m_position += m_forward * speed.z * deltaTime;
    m_viewMatrixRecalc = true;
}

void Camera::rotate(float angle, const glm::vec3& axis) {
    glm::vec3 n = glm::normalize(axis);
    glm::quat q = glm::angleAxis(angle, n);
    m_up = glm::normalize(glm::rotate(q, m_up));
    m_forward = glm::normalize(glm::rotate(q, m_forward));

    m_rotation = glm::normalize(q * m_rotation);

    m_viewMatrixRecalc = true;
}

void Camera::setViewByMouse(int x, int y, float dt) {
    float angleY = dt * static_cast<float>(x) * 2;
    float angleZ = dt * static_cast<float>(y) * 2;
    rotatePitch(-angleZ * dt);
    rotate(-angleY * dt, glm::vec3(0.0, 1.0, 0.0));
}

glm::mat4 Camera::getViewMatrix() {
    if (m_viewMatrixRecalc) {
        glm::quat q = m_rotation;
        q.x *= -1.0f;
        q.y *= -1.0f;
        q.z *= -1.0f;
        m_viewMatrix = glm::mat4_cast(q);

        glm::vec3 v = -m_position;
        glm::mat4 m = m_viewMatrix;
        m_viewMatrix[3] = (m[0] * v[0]) + (m[1] * v[1]) + (m[2] * v[2]) + m[3];

        m_viewMatrixRecalc = false;
    }

    return m_viewMatrix;
}

void Camera::translate(glm::vec3 position) {
    m_position = position;

    m_viewMatrixRecalc = true;
}

void Camera::lookAt(glm::vec3 const &center, const glm::vec3& up) {
    m_forward = glm::normalize(center - m_position);
    m_up = up;

    glm::mat3 m;
    m[0] = -1.0f * cross(m_up, m_forward);
    m[1] = m_up;
    m[2] = -1.0f * m_forward;
    m_rotation = glm::quat_cast(m);
    m_viewMatrixRecalc = true;
}


void Camera::rotatePitch(float angle) {
    glm::quat q = glm::angleAxis(angle, -cross(m_up, m_forward));

    m_up = glm::normalize(glm::rotate(q, m_up));
    m_forward = glm::normalize(glm::rotate(q, m_forward));

    m_rotation = glm::normalize(q * m_rotation);

    m_viewMatrixRecalc = true;
}

void Camera::rotateYaw(float angle) {
    glm::quat q = glm::angleAxis(angle, m_up);

    m_forward = glm::normalize(glm::rotate(q, m_forward));

    m_rotation = glm::normalize(q * m_rotation);

    m_viewMatrixRecalc = true;
}

void Camera::rotateRoll(float angle) {
    glm::quat q = glm::angleAxis(angle, m_forward);

    m_up = glm::normalize(glm::rotate(q, m_up));

    m_rotation = glm::normalize(q * m_rotation);

    m_viewMatrixRecalc = true;
}
