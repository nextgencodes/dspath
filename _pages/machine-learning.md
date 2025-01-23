---
layout: single
permalink: /machine-learning/
strip_title: true
---

# Machine Learning topics to explore
<ul>
    {% for post in site.courses %}
        {% if post.tags contains 'Machine Learning' %}
          <li>
            <a href="{{ post.url }}">{{ post.title }}</a>
          </li>
        {% endif %}
    {% endfor %}
</ul>
