package com.example.recipe;

import java.util.List;

record Recipe(String name, String description, List<String> ingredients, List<String> instructions, String imageUrl) {

    Recipe(Recipe recipe, String imageUrl) {
        this(recipe.name, recipe.description, recipe.ingredients, recipe.instructions, imageUrl);
    }
}