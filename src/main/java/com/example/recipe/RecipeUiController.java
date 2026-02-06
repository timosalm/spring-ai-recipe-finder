package com.example.recipe;

import io.github.bucket4j.Bucket;
import io.github.bucket4j.ConsumptionProbe;
import jakarta.servlet.http.HttpServletRequest;
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
    private final InputValidator inputValidator;
    private final RateLimitingConfig rateLimitingConfig;

    RecipeUiController(RecipeService recipeService, ChatModel chatModel, Optional<ImageModel> imageModel,
                      InputValidator inputValidator, RateLimitingConfig rateLimitingConfig) {
        this.recipeService = recipeService;
        this.chatModel = chatModel;
        this.imageModel = imageModel;
        this.inputValidator = inputValidator;
        this.rateLimitingConfig = rateLimitingConfig;
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
    String fetchRecipeUiFor(FetchRecipeData fetchRecipeData, Model model, HttpServletRequest request) throws Exception {
        try {
            // Rate limiting check
            String clientKey = getClientKey(request);
            Bucket bucket = rateLimitingConfig.resolveRecipeGenerationBucket(clientKey);
            ConsumptionProbe probe = bucket.tryConsumeAndReturnRemaining(1);
            
            if (!probe.isConsumed()) {
                long waitForRefill = probe.getNanosToWaitForRefill() / 1_000_000_000;
                log.warn("Rate limit exceeded for recipe generation from client: {}", clientKey);
                model.addAttribute("error", "Too many requests. Please wait " + waitForRefill + " seconds before trying again.");
                return fetchUI(model);
            }

            // Input validation
            inputValidator.validateIngredients(fetchRecipeData.ingredients());

            // Process the recipe request
            Recipe recipe;
            try {
                recipe = recipeService.fetchRecipeFor(fetchRecipeData.ingredients(), fetchRecipeData.isPreferAvailableIngredients(), fetchRecipeData.isPreferOwnRecipes());
            } catch (Exception e) {
                log.info("Retry RecipeUiController:fetchRecipeFor after exception caused by LLM");
                recipe = recipeService.fetchRecipeFor(fetchRecipeData.ingredients(), fetchRecipeData.isPreferAvailableIngredients(), fetchRecipeData.isPreferOwnRecipes());
            }
            
            model.addAttribute("recipe", recipe);
            model.addAttribute("fetchRecipeData", fetchRecipeData);
            log.info("Recipe generated successfully for client: {}", clientKey);
        } catch (IllegalArgumentException e) {
            log.warn("Invalid recipe request: {}", e.getMessage());
            model.addAttribute("error", e.getMessage());
        } catch (Exception e) {
            log.error("Error generating recipe", e);
            model.addAttribute("error", "An error occurred while generating the recipe. Please try again.");
        }
        
        return fetchUI(model);
    }

    /**
     * Extracts a client identifier for rate limiting (IP address).
     */
    private String getClientKey(HttpServletRequest request) {
        String xff = request.getHeader("X-Forwarded-For");
        if (xff != null && !xff.isEmpty()) {
            return xff.split(",")[0].trim();
        }
        return request.getRemoteAddr();
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
