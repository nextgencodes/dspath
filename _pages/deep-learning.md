---
layout: single
permalink: /deep-learning/
strip_title: true
---

# Deep Learning topics to explore
<ul>
    {% for post in site.courses %}
        {% if post.tags contains 'Deep Learning' %}
          <li>
            <a href="{{ post.url }}">{{ post.title }}</a>
          </li>
        {% endif %}
    {% endfor %}
</ul>
