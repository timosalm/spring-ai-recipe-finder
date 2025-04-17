package com.example.recipe;

import org.apache.commons.lang3.reflect.FieldUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.image.ImageModel;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.springframework.util.StringUtils.capitalize;

@Controller
@RequestMapping("/")
class RecipeUiController {

    private static final Logger log = LoggerFactory.getLogger(RecipeUiController.class);

    private final RecipeService recipeService;
    private final ChatModel chatModel;
    private final Optional<ImageModel> imageModel;

    RecipeUiController(RecipeService recipeService, ChatModel chatModel, Optional<ImageModel> imageModel) {
        this.recipeService = recipeService;
        this.chatModel = chatModel;
        this.imageModel = imageModel;
    }

    @GetMapping
    String fetchUI(Model model) {
        var aiModelNames = getAiModelNames();
        model.addAttribute("aiModel", String.join(" & ", aiModelNames));
        if (!model.containsAttribute("fetchRecipeData")) {
            model.addAttribute("fetchRecipeData", new FetchRecipeData());
        }
        return "index";
    }

    @PostMapping
    String fetchRecipeUiFor(FetchRecipeData fetchRecipeData, Model model) throws Exception {
        Recipe recipe;
        try {
            recipe = recipeService.fetchRecipeFor(fetchRecipeData.ingredients(), fetchRecipeData.isPreferAvailableIngredients(), fetchRecipeData.isPreferOwnRecipes());
        } catch (Exception e) {
            log.info("Retry RecipeUiController:fetchRecipeFor after exception caused by LLM");
            recipe = recipeService.fetchRecipeFor(fetchRecipeData.ingredients(), fetchRecipeData.isPreferAvailableIngredients(), fetchRecipeData.isPreferOwnRecipes());
        }
        model.addAttribute("recipe", recipe);
        model.addAttribute("fetchRecipeData", fetchRecipeData);
        return fetchUI(model);
    }

    private List<String> getAiModelNames() {
        var modelNames = new ArrayList<String>();
        var chatModelProvider = chatModel.getClass().getSimpleName().replace("ChatModel", "");
        var chatModelDefaultOptions = chatModel.getDefaultOptions();
        try {
            var modelName = (String)FieldUtils.readField(chatModelDefaultOptions, "model", true);
            modelNames.add("%s (%s)".formatted(chatModelProvider, capitalize(modelName)));
        } catch (Exception e1) {
            try {
                var modelName = (String)FieldUtils.readField(chatModelDefaultOptions, "deploymentName", true);
                modelNames.add("%s (%s)".formatted(chatModelProvider, capitalize(modelName)));
            } catch (Exception e2) {
                modelNames.add(chatModelProvider);
            }
        }

        if (imageModel.isPresent()) {
            var imageModelProvider = imageModel.get().getClass().getSimpleName().replace("ImageModel", "");
            try {
                var imageModelDefaultOptions = FieldUtils.readField(imageModel.get(), "defaultOptions", true);
                var imageModel = (String)FieldUtils.readField(imageModelDefaultOptions, "model", true);
                modelNames.add("%s (%s)".formatted(imageModelProvider, capitalize(imageModel)));
            } catch (Exception e) {
                modelNames.add(imageModelProvider);
            }
        }

        return modelNames;
    }
}
