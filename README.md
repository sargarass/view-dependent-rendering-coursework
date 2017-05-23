# view-dependent-rendering-coursework
Based on Carles Loop article: [Real-Time View-Dependent Rendering of Parametric Surfaces](http://research.microsoft.com/en-us/um/people/cloop/EisenEtAl2009.pdf)

#System Requirements:
GPU with cuda compute capability 2.0 and opengl 3.3 support with extensions: GL_EXT_direct_state_access, GL_NV_framebuffer_multisample_coverage

Libraries:
1. GLFW
2. GLEW
3. GLM
4. CUB

# Описание на русском:
В компьютерной графике популярным представлением для гладких поверхностей является параметрическое представление в виде набора патчей, например: в виде NURBS-патчей, патчей Безье и т.п. Это позволяет представить гладкий объект, не имеющий аналитического определения, в виде малого набора гладких патчей.

При рендеринге набор патчей аппроксимируется с помощью полигональных сеток (*polygonal mesh*), которые допускают простое  выполнение отсечения и закрашивания. Когда аппроксимация выполняется как препроцесс до запуска основной программы, получается статическая модель. При рассмотрении модели с ближнего расстояния видны артефакты и угловатости полигонального представления, исчезает имитация гладкости. Аналогично, при рассмотрении модели с дальнего расстояния много вычислений тратится впустую для отрисовки избыточных полигонов, без которых модель сохранила бы гладкость. Кроме того, сильно тесселированная модель требует большого объёма памяти (представление в виде патчей очень компактно) и затрачивает пропускную способность шины при передаче на GPU, а также тяжело анимируется. Более гибким подходом является передача патчей на GPU и аппроксимация на лету в зависимости от проекции.

Программная реализация алгоритма тесселяции гладких параметрических поверхностей на GPU была выполнена с использованием технологии параллельного программирования CUDA в рамках курсовой работы по компьютерной графике (3 курс ИУ-9 2015 год). Реализованный метод устраняет разрывы при визуализации, имеет высокую скорость работы, хорошо масштабируется с увеличением числа потоков/патчей в модели. 

На картинках можно видеть, что ближайшие к камере полигоны имеют наибольшую детализацию.
![]({{site.baseurl}}//%D0%9E%D0%B1%D1%8A%D0%B5%D0%BA%D1%8210.png)
![]({{site.baseurl}}//%D0%9E%D0%B1%D1%8A%D0%B5%D0%BA%D1%8225.png)![]({{site.baseurl}}//%D0%9E%D0%B1%D1%8A%D0%B5%D0%BA%D1%8212.png)

