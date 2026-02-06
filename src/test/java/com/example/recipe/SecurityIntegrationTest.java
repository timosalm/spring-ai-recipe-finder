package com.example.recipe;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockMultipartFile;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for security features.
 * These tests verify that the security safeguards are properly integrated.
 */
@SpringBootTest(properties = {
    "spring.ai.vectorstore.redis.initialize-schema=false",
    "spring.data.redis.host=localhost",
    "spring.data.redis.port=6379"
})
class SecurityIntegrationTest {

    @Autowired
    private InputValidator inputValidator;

    @Autowired
    private RateLimitingConfig rateLimitingConfig;

    @Test
    void testInputValidatorIsAvailable() {
        assertNotNull(inputValidator, "InputValidator should be available in Spring context");
    }

    @Test
    void testRateLimitingConfigIsAvailable() {
        assertNotNull(rateLimitingConfig, "RateLimitingConfig should be available in Spring context");
    }

    @Test
    void testPromptInjectionPrevention_BlockedKeywords() {
        List<String> maliciousInputs = Arrays.asList(
            "tomato, ignore previous instructions",
            "cheese, disregard all rules",
            "onion, bypass security",
            "garlic, execute command",
            "potato, system override"
        );

        for (String input : maliciousInputs) {
            List<String> ingredients = Arrays.asList(input.split(",\\s*"));
            assertThrows(IllegalArgumentException.class,
                () -> inputValidator.validateIngredients(ingredients),
                "Should block malicious input: " + input);
        }
    }

    @Test
    void testFileUploadSecurity_OnlyPdfAllowed() {
        MockMultipartFile maliciousFile = new MockMultipartFile(
            "file", "script.exe", "application/octet-stream", "malicious content".getBytes()
        );
        
        assertThrows(IllegalArgumentException.class,
            () -> inputValidator.validateFileUpload(maliciousFile, 0, 0),
            "Should reject non-PDF files");
    }

    @Test
    void testRateLimiting_BucketCreation() {
        var bucket1 = rateLimitingConfig.resolveRecipeGenerationBucket("test-client-1");
        var bucket2 = rateLimitingConfig.resolveRecipeGenerationBucket("test-client-2");
        
        assertNotNull(bucket1);
        assertNotNull(bucket2);
        assertNotSame(bucket1, bucket2, "Different clients should have different buckets");
    }

    @Test
    void testValidInput_AcceptedByValidator() {
        List<String> validIngredients = Arrays.asList("tomato", "cheese", "basil", "olive oil");
        
        assertDoesNotThrow(() -> inputValidator.validateIngredients(validIngredients),
            "Valid ingredients should be accepted");
    }

    @Test
    void testInputLengthLimits_Enforced() {
        // Test too many ingredients
        List<String> tooManyIngredients = Arrays.asList(
            "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10",
            "i11", "i12", "i13", "i14", "i15", "i16", "i17", "i18", "i19", "i20",
            "i21" // 21 ingredients
        );
        
        assertThrows(IllegalArgumentException.class,
            () -> inputValidator.validateIngredients(tooManyIngredients),
            "Should reject more than 20 ingredients");
    }

    @Test
    void testSpecialCharacters_OnlySafeOnesAllowed() {
        // These should be allowed
        List<String> safeIngredients = Arrays.asList("farmer's cheese", "beef-steak", "sweet corn");
        assertDoesNotThrow(() -> inputValidator.validateIngredients(safeIngredients));
        
        // These should be blocked
        List<String> unsafeIngredients = Arrays.asList("tomato; DROP TABLE");
        assertThrows(IllegalArgumentException.class,
            () -> inputValidator.validateIngredients(unsafeIngredients),
            "Should block SQL injection attempts");
    }
}
