---
layout: single
permalink: /embedded/
strip_title: true
---

# Embedded Programming topics to explore
<ul>
    {% for post in site.courses %}
        {% if post.tags contains 'Embedded' %}
          <li>
            <a href="{{ post.url }}">{{ post.title }}</a>
          </li>
        {% endif %}
    {% endfor %}
</ul>
