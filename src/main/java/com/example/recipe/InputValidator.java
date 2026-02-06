package com.example.recipe;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Validates and sanitizes user inputs to prevent prompt injection and security vulnerabilities.
 */
@Component
public class InputValidator {

    private static final Logger log = LoggerFactory.getLogger(InputValidator.class);

    // Maximum lengths for various inputs
    private static final int MAX_INGREDIENT_LENGTH = 100;
    private static final int MAX_INGREDIENTS_COUNT = 20;
    private static final int MAX_TOTAL_INPUT_LENGTH = 500;
    private static final long MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

    // Allowed characters pattern for ingredients (letters, spaces, hyphens, apostrophes)
    private static final Pattern SAFE_INGREDIENT_PATTERN = Pattern.compile("^[a-zA-Z0-9\\s\\-']+$");

    // Dangerous keywords that might indicate prompt injection attempts
    private static final Set<String> BLOCKED_KEYWORDS = Set.of(
            "ignore", "disregard", "forget", "system", "prompt", "instruction",
            "override", "bypass", "admin", "root", "sudo", "execute",
            "eval", "inject", "script", "command", "shell", "hack"
    );

    // Allowed file types for recipe documents
    private static final Set<String> ALLOWED_FILE_TYPES = Set.of(
            "application/pdf"
    );

    /**
     * Validates a list of ingredients for recipe generation.
     * 
     * @param ingredients List of ingredient strings to validate
     * @throws IllegalArgumentException if validation fails
     */
    public void validateIngredients(List<String> ingredients) {
        if (ingredients == null || ingredients.isEmpty()) {
            log.warn("Empty or null ingredients list provided");
            throw new IllegalArgumentException("Ingredients list cannot be empty");
        }

        if (ingredients.size() > MAX_INGREDIENTS_COUNT) {
            log.warn("Too many ingredients provided: {}", ingredients.size());
            throw new IllegalArgumentException("Too many ingredients. Maximum allowed: " + MAX_INGREDIENTS_COUNT);
        }

        int totalLength = 0;
        for (String ingredient : ingredients) {
            if (ingredient == null || ingredient.trim().isEmpty()) {
                log.warn("Empty ingredient detected");
                throw new IllegalArgumentException("Ingredients cannot be empty");
            }

            String trimmed = ingredient.trim();
            
            if (trimmed.length() > MAX_INGREDIENT_LENGTH) {
                log.warn("Ingredient too long: {} characters", trimmed.length());
                throw new IllegalArgumentException("Ingredient too long. Maximum length: " + MAX_INGREDIENT_LENGTH);
            }

            totalLength += trimmed.length();

            if (!SAFE_INGREDIENT_PATTERN.matcher(trimmed).matches()) {
                log.warn("Invalid characters in ingredient: {}", sanitizeForLogging(trimmed));
                throw new IllegalArgumentException("Invalid characters in ingredient. Only letters, numbers, spaces, hyphens, and apostrophes are allowed.");
            }

            String lowerCase = trimmed.toLowerCase();
            for (String blockedKeyword : BLOCKED_KEYWORDS) {
                if (lowerCase.contains(blockedKeyword)) {
                    log.warn("Blocked keyword detected in ingredient: {}", blockedKeyword);
                    throw new IllegalArgumentException("Invalid ingredient content detected");
                }
            }
        }

        if (totalLength > MAX_TOTAL_INPUT_LENGTH) {
            log.warn("Total input length too long: {} characters", totalLength);
            throw new IllegalArgumentException("Total input too long. Maximum: " + MAX_TOTAL_INPUT_LENGTH + " characters");
        }
    }

    /**
     * Validates an uploaded file for recipe documents.
     * 
     * @param file The multipart file to validate
     * @param pageTopMargin Top margin value
     * @param pageBottomMargin Bottom margin value
     * @throws IllegalArgumentException if validation fails
     */
    public void validateFileUpload(MultipartFile file, int pageTopMargin, int pageBottomMargin) {
        if (file == null || file.isEmpty()) {
            log.warn("Empty or null file upload attempted");
            throw new IllegalArgumentException("File cannot be empty");
        }

        // Validate file size
        if (file.getSize() > MAX_FILE_SIZE) {
            log.warn("File too large: {} bytes", file.getSize());
            throw new IllegalArgumentException("File too large. Maximum size: " + (MAX_FILE_SIZE / 1024 / 1024) + "MB");
        }

        // Validate file type
        String contentType = file.getContentType();
        if (contentType == null || !ALLOWED_FILE_TYPES.contains(contentType)) {
            log.warn("Invalid file type: {}", contentType);
            throw new IllegalArgumentException("Invalid file type. Only PDF files are allowed.");
        }

        // Validate filename
        String filename = file.getOriginalFilename();
        if (filename == null || filename.trim().isEmpty()) {
            log.warn("Missing filename");
            throw new IllegalArgumentException("Filename is required");
        }

        if (!filename.toLowerCase().endsWith(".pdf")) {
            log.warn("Invalid file extension: {}", filename);
            throw new IllegalArgumentException("Only PDF files are allowed");
        }

        // Validate margin parameters
        if (pageTopMargin < 0 || pageTopMargin > 1000) {
            log.warn("Invalid top margin: {}", pageTopMargin);
            throw new IllegalArgumentException("Invalid page top margin value");
        }

        if (pageBottomMargin < 0 || pageBottomMargin > 1000) {
            log.warn("Invalid bottom margin: {}", pageBottomMargin);
            throw new IllegalArgumentException("Invalid page bottom margin value");
        }

        log.info("File upload validation successful: {} ({} bytes)", sanitizeForLogging(filename), file.getSize());
    }

    /**
     * Sanitizes a string for safe logging by removing potentially harmful characters.
     * 
     * @param input The input string
     * @return Sanitized string safe for logging
     */
    private String sanitizeForLogging(String input) {
        if (input == null) {
            return "null";
        }
        // Remove control characters and limit length for logging
        return input.replaceAll("[\\p{Cntrl}]", "").substring(0, Math.min(input.length(), 100));
    }
}
